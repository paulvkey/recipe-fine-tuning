#!/usr/bin/env python3
"""汇总官方 BLEU/ROUGE 结果并补充食谱输出结构质量指标。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


THINK_PATTERN = re.compile(r"</?think>|思考过程|推理过程", re.IGNORECASE)
STEP_PATTERN = re.compile(r"(?m)^\s*(?:\d+[.、)]|第[一二三四五六七八九十百\d]+步)")
INGREDIENT_PATTERN = re.compile(r"(?:^|\n)\s*(?:食材|原料|材料)\s*[：:]", re.MULTILINE)
INSTRUCTION_PATTERN = re.compile(r"(?:^|\n)\s*(?:制作步骤|做法|步骤)\s*[：:]", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总各 checkpoint 的官方指标与食谱结构指标。")
    parser.add_argument("result_root", type=Path, help="evaluate_checkpoints.sh 的结果目录")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def prediction_stats(path: Path) -> dict[str, float | int]:
    count = empty = formatted = numbered = thinking = 0
    length_ratios: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            prediction = row.get("predict", "")
            label = row.get("label", "")
            prediction = prediction if isinstance(prediction, str) else str(prediction)
            label = label if isinstance(label, str) else str(label)
            count += 1
            empty += not prediction.strip()
            formatted += bool(INGREDIENT_PATTERN.search(prediction) and INSTRUCTION_PATTERN.search(prediction))
            numbered += bool(STEP_PATTERN.search(prediction))
            thinking += bool(THINK_PATTERN.search(prediction))
            length_ratios.append(len(prediction) / max(len(label), 1))

    if count == 0:
        raise ValueError(f"预测文件为空：{path}")
    return {
        "samples": count,
        "empty_rate": round(empty * 100 / count, 4),
        "recipe_format_rate": round(formatted * 100 / count, 4),
        "numbered_steps_rate": round(numbered * 100 / count, 4),
        "thinking_leak_rate": round(thinking * 100 / count, 4),
        "mean_length_ratio": round(statistics.fmean(length_ratios), 4),
        "median_length_ratio": round(statistics.median(length_ratios), 4),
    }


def checkpoint_step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else -1


def is_candidate(row: dict[str, Any]) -> bool:
    return row["step"] >= 0 or row["name"] == "final_model"


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        predictions_path = run_dir / "generated_predictions.jsonl"
        if not predictions_path.is_file():
            print(f"警告：缺少预测文件，跳过 {run_dir}", file=sys.stderr)
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        stats = prediction_stats(predictions_path)
        rows.append(
            {
                "name": run_dir.name,
                "checkpoint_path": metrics.get("checkpoint_path", ""),
                "step": checkpoint_step(run_dir.name),
                "bleu_4": round(safe_float(metrics.get("predict_bleu-4")), 4),
                "rouge_1": round(safe_float(metrics.get("predict_rouge-1")), 4),
                "rouge_2": round(safe_float(metrics.get("predict_rouge-2")), 4),
                "rouge_l": round(safe_float(metrics.get("predict_rouge-l")), 4),
                "samples_per_second": round(safe_float(metrics.get("predict_samples_per_second")), 4),
                **stats,
            }
        )
    return rows


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    # 文本参考指标优先；格式合规率用于同分时消除明显的坏输出。
    return (
        row["rouge_l"],
        row["rouge_2"],
        row["recipe_format_rate"],
        row["bleu_4"],
    )


def main() -> int:
    args = parse_args()
    root = args.result_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"找不到评估结果目录：{root}")
    rows = collect(root)
    if not rows:
        raise ValueError(f"没有找到可汇总的 metrics.json：{root}")

    checkpoint_rows = sorted(
        (row for row in rows if is_candidate(row)),
        key=rank_key,
        reverse=True,
    )
    baseline_rows = [row for row in rows if not is_candidate(row)]
    ordered = checkpoint_rows + baseline_rows

    json_path = root / "checkpoint_ranking.json"
    csv_path = root / "checkpoint_ranking.csv"
    report_path = root / "evaluation_report.md"
    best_path = root / "best_checkpoint.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    columns = list(ordered[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)

    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Checkpoint 最终评估报告\n\n")
        handle.write(
            "排序规则：ROUGE-L、ROUGE-2、食谱格式合规率、BLEU-4 依次降序。"
            "BLEU/ROUGE 是参考答案重合度指标，不能单独代表事实正确性或口味质量。\n\n"
        )
        handle.write(
            "| 排名 | 名称 | ROUGE-L | ROUGE-2 | BLEU-4 | 食谱格式率 | 编号步骤率 | 思考泄漏率 | 长度比 |\n"
        )
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        rank = 0
        for row in ordered:
            if is_candidate(row):
                rank += 1
                rank_text = str(rank)
            else:
                rank_text = "基线"
            handle.write(
                f"| {rank_text} | {row['name']} | {row['rouge_l']:.4f} | "
                f"{row['rouge_2']:.4f} | {row['bleu_4']:.4f} | "
                f"{row['recipe_format_rate']:.2f}% | {row['numbered_steps_rate']:.2f}% | "
                f"{row['thinking_leak_rate']:.2f}% | {row['median_length_ratio']:.3f} |\n"
            )
        handle.write(
            "\n建议对排名前 3 的 checkpoint 再人工检查同一批代表性样例，重点核对食材遗漏、"
            "用量合理性、步骤可执行性和食品安全，再确定部署版本。\n"
        )

    if checkpoint_rows:
        best = checkpoint_rows[0]
        with best_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(str(best["checkpoint_path"] or best["name"]) + "\n")
        print(f"最佳 checkpoint：{best['name']}（ROUGE-L={best['rouge_l']:.4f}）")
    else:
        print("警告：只有基础模型结果，没有 checkpoint 可排序", file=sys.stderr)
    print(f"评估报告：{report_path}")
    print(f"排序明细：{csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
