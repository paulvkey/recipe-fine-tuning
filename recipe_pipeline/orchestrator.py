#!/usr/bin/env python3
"""One command for sample-driven rule generation and full recipe cleaning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from recipe_pipeline import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT_ROOT / "data/base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m recipe_pipeline",
        description="一键执行采样建库、可选审计和完整清洗转换。",
    )
    parser.add_argument("input", type=Path, help="原始食谱 JSONL 或 JSONL.GZ")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline_output"))
    parser.add_argument("--sample-size", type=int, default=100_000, help="规则建库随机样本数，默认 100000")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--min-typo-count", type=int, default=3, help="自动启用纠错的最低样本频次，默认 3")
    parser.add_argument("--min-protected-count", type=int, default=10, help="菜名保护词最低样本频次，默认 10")
    parser.add_argument(
        "--min-ingredient-protected-count",
        type=int,
        default=30,
        help="食材保护词最低样本频次，默认 30",
    )
    parser.add_argument("--with-audit", action="store_true", help="清洗前额外执行独立全量审计（会多扫描一遍）")
    parser.add_argument("--target-count", type=int, help="最终随机输出的有效样本数；默认输出全部")
    parser.add_argument("--selection-seed", type=int, default=20260722, help="最终样本随机选择种子")
    parser.add_argument("--scan-limit", type=int, help="测试用：限制所有阶段处理的原始记录数")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    if args.target_count is not None and args.target_count <= 0:
        parser.error("--target-count 必须大于 0")
    if min(
        args.sample_size,
        args.min_typo_count,
        args.min_protected_count,
        args.min_ingredient_protected_count,
    ) <= 0:
        parser.error("采样数量和各频次阈值必须大于 0")
    return args


def banner(step: int, total: int, title: str) -> None:
    border = "=" * 78
    print(f"\n{border}\nSTEP {step}/{total}  {title}\n{border}", file=sys.stderr)


def run(command: list[str], step: int, total: int, title: str) -> None:
    banner(step, total, title)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    config_dir = output_dir / "generated_config"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    total_steps = 4 if args.with_audit else 3
    current_step = 1

    bootstrap_command = [
        python,
        "-m", "recipe_pipeline.bootstrap",
        str(input_path),
        str(config_dir),
        "--sample-size", str(args.sample_size),
        "--seed", str(args.seed),
        "--min-typo-count", str(args.min_typo_count),
        "--min-protected-count", str(args.min_protected_count),
        "--min-ingredient-protected-count", str(args.min_ingredient_protected_count),
        "--progress-every", str(args.progress_every),
    ]
    if args.scan_limit:
        bootstrap_command.extend(["--scan-limit", str(args.scan_limit)])
    run(bootstrap_command, current_step, total_steps, "随机采样并生成纠错规则与保护词")
    current_step += 1

    generated_rules = config_dir / "typo_rules.generated.json"
    generated_words = config_dir / "protected_words.generated.txt"
    if args.with_audit:
        audit_command = [
            python,
            "-m", "recipe_pipeline.audit",
            str(input_path),
            str(output_dir / "audit_report.json"),
            "--typo-rules", str(generated_rules),
            "--noise-rules", str(BASE_DATA / "noise_rules.json"),
            "--protected-words", str(generated_words),
            "--progress-every", str(args.progress_every),
        ]
        if args.scan_limit:
            audit_command.extend(["--limit", str(args.scan_limit)])
        run(audit_command, current_step, total_steps, "独立全量审计")
        current_step += 1

    clean_command = [
        python,
        "-m", "recipe_pipeline.clean",
        str(input_path),
        str(output_dir / "recipe_train_clean.jsonl"),
        "--review", str(output_dir / "recipe_review.jsonl"),
        "--rejected", str(output_dir / "recipe_rejected.jsonl"),
        "--report", str(output_dir / "clean_report.json"),
        "--typo-rules", str(generated_rules),
        "--noise-rules", str(BASE_DATA / "noise_rules.json"),
        "--protected-words", str(generated_words),
        "--progress-every", str(args.progress_every),
    ]
    if args.target_count:
        clean_command.extend(["--target-count", str(args.target_count)])
    clean_command.extend(["--selection-seed", str(args.selection_seed)])
    if args.scan_limit:
        clean_command.extend(["--scan-limit", str(args.scan_limit)])
    run(clean_command, current_step, total_steps, "清洗、风险分流、去重与随机定量转换")
    current_step += 1

    manifest = {
        "pipeline_version": __version__,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "generated_config": {
            "typo_rules": str(generated_rules),
            "protected_words": str(generated_words),
            "bootstrap_report": str(config_dir / "bootstrap_report.json"),
        },
        "outputs": {
            "train": str(output_dir / "recipe_train_clean.jsonl"),
            "review": str(output_dir / "recipe_review.jsonl"),
            "rejected": str(output_dir / "recipe_rejected.jsonl"),
            "report": str(output_dir / "clean_report.json"),
        },
        "requested_target_count": args.target_count,
        "selection_seed": args.selection_seed if args.target_count is not None else None,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    banner(current_step, total_steps, "执行完成")
    with (output_dir / "clean_report.json").open(encoding="utf-8") as source:
        summary = json.load(source).get("summary", {})
    print(
        f"有效候选：{summary.get('eligible', 0):,}\n"
        f"最终输出：{summary.get('clean', 0):,}\n"
        f"进入审核：{summary.get('review', 0):,}\n"
        f"拒绝数据：{summary.get('rejected', 0):,}\n"
        f"精确重复：{summary.get('duplicates', 0):,}\n"
        f"训练文件：{output_dir / 'recipe_train_clean.jsonl'}\n"
        f"统计报告：{output_dir / 'clean_report.json'}\n"
        f"输出目录：{output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
