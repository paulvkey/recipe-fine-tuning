#!/usr/bin/env python3
"""从清洗后全量数据中生成与训练样本严格去重的固定评估集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


FIELDS = ("instruction", "input", "output")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="流式扫描全量清洗数据，排除训练集后用蓄水池算法生成固定评估集。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "pipeline_output" / "recipe_train_clean.jsonl",
        help="清洗后的全量 JSONL",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "training_sample" / "recipe_train_sample_100000.jsonl",
        help="必须从候选中排除的训练 JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "data" / "recipe_eval_holdout.jsonl",
        help="评估集输出路径",
    )
    parser.add_argument("--count", type=int, default=1000, help="评估样本数，默认 1000")
    parser.add_argument("--seed", type=int, default=20260811, help="随机种子")
    parser.add_argument(
        "--report",
        type=Path,
        default=SCRIPT_DIR / "data" / "eval_holdout_report.json",
        help="采样报告输出路径",
    )
    return parser.parse_args()


def normalized_record(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(value.get(field), str) for field in FIELDS):
        return None
    record = {field: value[field].strip() for field in FIELDS}
    if not record["instruction"] or not record["output"]:
        return None
    return record


def record_digest(record: dict[str, str]) -> bytes:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def load_train_digests(path: Path) -> tuple[set[bytes], int]:
    digests: set[bytes] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = normalized_record(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"训练数据第 {line_no} 行不是合法 JSON：{exc}") from exc
            if record is None:
                raise ValueError(f"训练数据第 {line_no} 行缺少有效的 instruction/input/output")
            rows += 1
            digests.add(record_digest(record))
    return digests, rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count 必须大于 0")
    if not args.source.is_file():
        raise FileNotFoundError(f"找不到全量清洗数据：{args.source}")
    if not args.train.is_file():
        raise FileNotFoundError(f"找不到训练样本：{args.train}")

    print("[1/3] 读取训练集指纹，用于严格排除数据泄漏", flush=True)
    train_digests, train_rows = load_train_digests(args.train)
    print(f"      训练记录 {train_rows:,} 条，唯一指纹 {len(train_digests):,} 个", flush=True)

    print("[2/3] 流式扫描全量清洗数据并执行蓄水池随机采样", flush=True)
    rng = random.Random(args.seed)
    reservoir: list[dict[str, str]] = []
    source_digests: set[bytes] = set()
    total_rows = malformed_rows = train_overlap_rows = duplicate_rows = eligible_rows = 0

    with args.source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                record = normalized_record(json.loads(line))
            except json.JSONDecodeError:
                record = None
            if record is None:
                malformed_rows += 1
                continue

            digest = record_digest(record)
            if digest in train_digests:
                train_overlap_rows += 1
                continue
            if digest in source_digests:
                duplicate_rows += 1
                continue
            source_digests.add(digest)

            eligible_rows += 1
            if len(reservoir) < args.count:
                reservoir.append(record)
            else:
                replacement = rng.randrange(eligible_rows)
                if replacement < args.count:
                    reservoir[replacement] = record

            if total_rows % 200_000 == 0:
                print(
                    f"      已扫描 {total_rows:,} 条，可用且唯一 {eligible_rows:,} 条",
                    flush=True,
                )

    if len(reservoir) < args.count:
        raise ValueError(
            f"排除训练集和无效数据后只有 {len(reservoir):,} 条，无法采样 {args.count:,} 条"
        )

    # 固定输出顺序，保证相同输入、count 和 seed 得到字节级可复现的文件。
    reservoir.sort(key=lambda item: record_digest(item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    print("[3/3] 写入评估集和可复现报告", flush=True)
    with temp_output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in reservoir:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_output.replace(args.output)

    report = {
        "source_file": str(args.source.resolve()),
        "train_file": str(args.train.resolve()),
        "output_file": str(args.output.resolve()),
        "seed": args.seed,
        "requested_count": args.count,
        "written_count": len(reservoir),
        "train_rows": train_rows,
        "train_unique_records": len(train_digests),
        "source_rows": total_rows,
        "malformed_or_invalid_rows": malformed_rows,
        "excluded_train_overlap_rows": train_overlap_rows,
        "excluded_duplicate_source_rows": duplicate_rows,
        "eligible_unique_rows": eligible_rows,
        "output_sha256": file_sha256(args.output),
        "disjoint_rule": "BLAKE2b-128 over normalized instruction/input/output",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp_report = args.report.with_name(args.report.name + ".tmp")
    with temp_report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_report.replace(args.report)

    print(f"完成：{args.output}（{len(reservoir):,} 条）")
    print(f"报告：{args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
