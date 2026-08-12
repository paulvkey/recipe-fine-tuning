#!/usr/bin/env python3
"""记录并汇总 LlamaFactory 后台训练状态，自动归类常见失败原因。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATUS_FILE = SCRIPT_DIR / "logs" / "train_status.json"
DEFAULT_PID_FILE = SCRIPT_DIR / "logs" / "train.pid"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "qwen3-8b-base" / "recipe-lora-sft"
TAIL_BYTES = 8 * 1024 * 1024

PROGRESS_PATTERN = re.compile(r"(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)")
LOSS_PATTERN = re.compile(
    r"\{'loss':\s*'([^']+)'[^\r\n]*?'learning_rate':\s*'([^']+)'"
    r"[^\r\n]*?'epoch':\s*'([^']+)'"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一条命令查看训练状态、最后进度、checkpoint 和自动错误诊断。"
    )
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")

    # 下面的参数由 train.sh 内部使用，不属于日常命令接口。
    parser.add_argument(
        "--update-state",
        choices=("RUNNING", "COMPLETED", "FAILED"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--exit-code", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(args: argparse.Namespace) -> int:
    path = args.status_file.expanduser().resolve()
    status = load_json(path)
    state = args.update_state
    assert state is not None

    if state == "RUNNING":
        status = {
            "state": state,
            "pid": args.pid,
            "exit_code": None,
            "started_at": now_iso(),
            "ended_at": None,
            "log_file": str(args.log_file.expanduser().resolve()) if args.log_file else None,
            "output_dir": str(args.output_dir.expanduser().resolve()) if args.output_dir else None,
        }
    else:
        status["state"] = state
        status["pid"] = args.pid or status.get("pid")
        status["exit_code"] = args.exit_code
        status["ended_at"] = now_iso()
        if args.log_file:
            status["log_file"] = str(args.log_file.expanduser().resolve())
        if args.output_dir:
            status["output_dir"] = str(args.output_dir.expanduser().resolve())
    atomic_write_json(path, status)
    return 0


def pid_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False

    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        # Linux 中僵尸进程仍能通过 kill(pid, 0)，需显式排除 Z 状态。
        fields = proc_stat.read_text(encoding="utf-8").split()
        return len(fields) < 3 or fields[2] != "Z"
    except OSError:
        return True


def read_pid_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value.isdigit() else None
    except OSError:
        return None


def latest_log(log_dir: Path) -> Path | None:
    try:
        candidates = [path for path in log_dir.glob("train_*.log") if path.is_file()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    except OSError:
        return None


def read_tail(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    with path.open("rb") as source:
        size = source.seek(0, os.SEEK_END)
        source.seek(max(0, size - TAIL_BYTES))
        return source.read().decode("utf-8", errors="replace")


def last_matching_line(text: str, patterns: tuple[str, ...]) -> str | None:
    lowered = tuple(pattern.lower() for pattern in patterns)
    for line in reversed(text.replace("\r", "\n").splitlines()):
        if any(pattern in line.lower() for pattern in lowered):
            return line.strip()[:1200]
    return None


def diagnose(text: str) -> tuple[str | None, str | None, list[str]]:
    checks = (
        (
            "CUDA 显存溢出",
            ("torch.outofmemoryerror", "cuda out of memory", "memory mapping failed with oom"),
            [
                "默认配置已启用梯度检查点；若仍然 OOM，把 TRAIN_BATCH_SIZE 调为 2、"
                "GRADIENT_ACCUMULATION_STEPS 调为 8。",
                "执行 nvidia-smi，确认没有其他进程占用训练 GPU。",
            ],
        ),
        (
            "磁盘空间不足",
            ("no space left on device", "disk quota exceeded"),
            ["执行 df -h 和 du -sh 检查模型、checkpoint 与日志所在磁盘。"],
        ),
        (
            "NCCL/GPU 通信错误",
            ("nccl error", "ncclerror", "unhandled system error"),
            ["单卡训练确认 GPU_IDS 只包含一个编号，并检查驱动、CUDA 与 GPU 健康状态。"],
        ),
        (
            "训练数值异常",
            ("loss': 'nan", 'loss": "nan', "grad_norm': 'nan", "nan loss"),
            ["检查异常样本、学习率和最近一次正常 loss；不要直接从产生 NaN 后的节点恢复。"],
        ),
        (
            "进程被系统终止",
            ("killed", "sigkill", "signal 9"),
            ["检查系统内存、作业时限和调度器日志；必要时联系服务器管理员。"],
        ),
        (
            "数据或 JSON 解析错误",
            ("jsondecodeerror", "error when tokenizing data", "failed to read data"),
            ["检查报错行附近的 JSONL，并确认 instruction/input/output 都是字符串。"],
        ),
        (
            "Python 运行异常",
            ("traceback (most recent call last)", "modulenotfounderror:"),
            ["关键异常已在下方提取；如信息不足，可查看日志末尾 80 行。"],
        ),
        (
            "启动或配置检查失败",
            ("错误：", "error: 找不到", "permission denied"),
            ["根据关键错误修正路径、权限、依赖或参数后重新执行 train.sh。"],
        ),
    )
    lowered = text.lower()
    for category, patterns, suggestions in checks:
        if any(pattern in lowered for pattern in patterns):
            detail = last_matching_line(text, patterns)
            if category == "Python 运行异常":
                detail = last_matching_line(
                    text,
                    ("error:", "exception:", "runtimeerror:", "valueerror:", "typeerror:"),
                ) or detail
            return category, detail, suggestions
    return None, None, []


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def collect_report(args: argparse.Namespace) -> dict[str, Any]:
    status_file = args.status_file.expanduser().resolve()
    status = load_json(status_file)
    pid = status.get("pid") or read_pid_file(args.pid_file.expanduser())
    alive = pid_is_alive(pid)

    recorded_state = status.get("state")
    if recorded_state == "RUNNING":
        state = "RUNNING" if alive else "UNEXPECTED_STOP"
    elif recorded_state in {"COMPLETED", "FAILED"}:
        state = recorded_state
    elif alive:
        state = "RUNNING"
    else:
        state = recorded_state or "UNKNOWN"

    if args.log_file:
        log_file = args.log_file.expanduser().resolve()
    elif status.get("log_file"):
        log_file = Path(status["log_file"]).expanduser().resolve()
    else:
        log_file = latest_log(status_file.parent)

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    elif status.get("output_dir"):
        output_dir = Path(status["output_dir"]).expanduser().resolve()
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    tail = read_tail(log_file)
    progress_matches = list(PROGRESS_PATTERN.finditer(tail))
    progress = None
    if progress_matches:
        match = progress_matches[-1]
        progress = {
            "percent": int(match.group(1)),
            "step": int(match.group(2)),
            "total_steps": int(match.group(3)),
        }

    loss_matches = list(LOSS_PATTERN.finditer(tail))
    metrics = None
    if loss_matches:
        match = loss_matches[-1]
        metrics = {
            "loss": match.group(1),
            "learning_rate": match.group(2),
            "epoch": match.group(3),
        }

    checkpoints = []
    if output_dir.is_dir():
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )

    error_category, error_detail, suggestions = diagnose(tail)
    if state == "UNKNOWN" and error_category:
        state = "FAILED"
    elif state == "UNKNOWN" and "train_runtime" in tail and "train_loss" in tail:
        state = "COMPLETED"
    return {
        "state": state,
        "recorded_state": recorded_state,
        "pid": pid,
        "pid_alive": alive,
        "exit_code": status.get("exit_code"),
        "started_at": status.get("started_at"),
        "ended_at": status.get("ended_at"),
        "status_file": str(status_file),
        "log_file": str(log_file) if log_file else None,
        "output_dir": str(output_dir),
        "progress": progress,
        "metrics": metrics,
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "error_category": error_category,
        "error_detail": error_detail,
        "suggestions": suggestions,
    }


def print_report(report: dict[str, Any]) -> None:
    labels = {
        "RUNNING": "运行中",
        "COMPLETED": "已正常完成",
        "FAILED": "异常退出",
        "UNEXPECTED_STOP": "进程已消失（异常或被外部终止）",
        "UNKNOWN": "暂无可用状态",
    }
    print("========== 训练状态诊断 ==========")
    print(f"状态：{labels.get(report['state'], report['state'])}")
    print(f"PID：{report['pid'] or '未知'}")
    if report["exit_code"] is not None:
        print(f"退出码：{report['exit_code']}")
    if report["started_at"]:
        print(f"开始时间：{report['started_at']}")
    if report["ended_at"]:
        print(f"结束时间：{report['ended_at']}")
    print(f"日志：{report['log_file'] or '未找到'}")
    print(f"输出目录：{report['output_dir']}")

    progress = report["progress"]
    if progress:
        print(
            f"最后进度：{progress['step']}/{progress['total_steps']} "
            f"({progress['percent']}%)"
        )
    metrics = report["metrics"]
    if metrics:
        print(
            f"最近指标：loss={metrics['loss']}，epoch={metrics['epoch']}，"
            f"learning_rate={metrics['learning_rate']}"
        )
    print(f"Checkpoint 数量：{report['checkpoint_count']}")
    print(f"最近 Checkpoint：{report['latest_checkpoint'] or '尚未生成'}")

    if report["error_category"]:
        print("\n---------- 自动错误诊断 ----------")
        print(f"错误类型：{report['error_category']}")
        if report["error_detail"]:
            print(f"关键错误：{report['error_detail']}")
        for suggestion in report["suggestions"]:
            print(f"建议：{suggestion}")
    elif report["state"] in {"FAILED", "UNEXPECTED_STOP"}:
        print("\n未识别出常见错误，请执行：")
        print(f"tail -n 80 {json.dumps(report['log_file'], ensure_ascii=False)}")
    print("==================================")


def main() -> int:
    args = parse_args()
    if args.update_state:
        return update_status(args)

    report = collect_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    if report["state"] in {"FAILED", "UNEXPECTED_STOP"}:
        return 2
    if report["state"] == "UNKNOWN":
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"状态检查失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
