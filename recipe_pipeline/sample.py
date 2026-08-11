#!/usr/bin/env python3
"""Uniformly sample a cleaned training JSONL without loading it into memory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import IO, Any

from recipe_pipeline import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "pipeline_output/recipe_train_clean.jsonl"


def open_text(path: Path, mode: str) -> IO[str]:
    """Open plain JSONL or gzip-compressed JSONL as UTF-8 text."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从清洗完成的训练 JSONL 中低内存、无偏地随机抽取指定数量的数据。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"清洗后的 JSONL/JSONL.GZ；默认 {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_sample_output"),
        help="独立输出目录；默认 training_sample_output",
    )
    parser.add_argument("--sample-size", type=int, default=100_000, help="抽取数量；默认 100000")
    parser.add_argument("--seed", type=int, default=20260810, help="随机种子；默认 20260810")
    parser.add_argument(
        "--output-name",
        help="样本文件名；默认 recipe_train_sample_<数量>.jsonl，可使用 .jsonl.gz",
    )
    parser.add_argument("--progress-every", type=int, default=100_000, help="每处理多少条显示一次进度")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已经存在的样本和报告")
    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size 必须大于 0")
    if args.progress_every < 0:
        parser.error("--progress-every 不能小于 0")
    if args.output_name is not None:
        output_name = Path(args.output_name)
        if output_name.name != args.output_name or not (
            args.output_name.endswith(".jsonl") or args.output_name.endswith(".jsonl.gz")
        ):
            parser.error("--output-name 必须是以 .jsonl 或 .jsonl.gz 结尾的单独文件名")
    return args


def banner(step: int, total: int, title: str) -> None:
    border = "=" * 72
    print(f"\n{border}\n采样步骤 {step}/{total}  {title}\n{border}", file=sys.stderr)


def count_records(source_path: Path, progress_every: int) -> tuple[int, int, int]:
    """Count non-empty records in one sequential, constant-memory pass."""
    total_lines = 0
    records = 0
    blank_lines = 0
    with open_text(source_path, "r") as source:
        for total_lines, line in enumerate(source, 1):
            if line.strip():
                records += 1
            else:
                blank_lines += 1
            if progress_every and total_lines % progress_every == 0:
                print(
                    f"  [统计进度] 已读取 {total_lines:,} 行，有效非空记录 {records:,}",
                    file=sys.stderr,
                )
    return total_lines, records, blank_lines


def validate_training_record(line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"源文件第 {line_number} 行不是有效 JSON：{error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"源文件第 {line_number} 行不是 JSON 对象")
    for field in ("instruction", "input", "output"):
        if field not in record:
            raise ValueError(f"源文件第 {line_number} 行缺少字段 {field!r}")
        if not isinstance(record[field], str):
            raise ValueError(f"源文件第 {line_number} 行的字段 {field!r} 不是字符串")
    return record


def sample_records(
    source_path: Path,
    temporary_output: Path,
    population: int,
    sample_size: int,
    seed: int,
    progress_every: int,
) -> tuple[int, str]:
    """Choose exactly N records uniformly while retaining source order."""
    needed = min(sample_size, population)
    expected = needed
    remaining = population
    selected = 0
    records_seen = 0
    rng = random.Random(seed)
    digest = hashlib.sha256()

    with open_text(source_path, "r") as source, open_text(temporary_output, "w") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            records_seen += 1
            choose = needed > 0 and rng.randrange(remaining) < needed
            if choose:
                # The source is already cleaned, so validating only selected rows
                # catches output corruption without reparsing the complete 1.5 GB file.
                validate_training_record(line, line_number)
                normalized_line = line.rstrip("\r\n") + "\n"
                target.write(normalized_line)
                digest.update(normalized_line.encode("utf-8"))
                selected += 1
                needed -= 1
            remaining -= 1
            if progress_every and records_seen % progress_every == 0:
                print(
                    f"  [抽取进度] 已扫描 {records_seen:,}/{population:,}，已选 {selected:,}/{expected:,}",
                    file=sys.stderr,
                )

    if records_seen != population or remaining != 0:
        raise RuntimeError("两次扫描得到的记录数不一致，源文件可能在采样期间发生了变化")
    if selected != expected or needed != 0:
        raise RuntimeError(f"采样数量异常：期望 {expected} 条，实际 {selected} 条")
    return selected, digest.hexdigest()


def main() -> int:
    args = parse_args()
    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_name = args.output_name or f"recipe_train_sample_{args.sample_size}.jsonl"
    output_path = output_dir / output_name
    report_path = output_dir / "sample_report.json"

    if not source_path.is_file():
        print(f"错误：找不到清洗数据文件：{source_path}", file=sys.stderr)
        return 2
    if source_path in (output_path.resolve(), report_path.resolve()):
        print("错误：输入文件不能同时作为输出文件或报告文件", file=sys.stderr)
        return 2
    if not args.overwrite:
        existing = [path for path in (output_path, report_path) if path.exists()]
        if existing:
            print(
                "错误：以下输出已存在，请更换目录或增加 --overwrite：\n  "
                + "\n  ".join(str(path) for path in existing),
                file=sys.stderr,
            )
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = source_path.stat()

    banner(1, 3, "统计清洗数据总量")
    total_lines, population, blank_lines = count_records(source_path, args.progress_every)
    if population == 0:
        print("错误：输入文件中没有可供采样的非空记录", file=sys.stderr)
        return 2
    source_after_count = source_path.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (
        source_after_count.st_size,
        source_after_count.st_mtime_ns,
    ):
        print("错误：源文件在统计期间发生了变化，请停止写入后重试", file=sys.stderr)
        return 2

    requested = args.sample_size
    actual_target = min(requested, population)
    if population < requested:
        print(
            f"  [采样提示] 仅有 {population:,} 条记录，少于请求的 {requested:,} 条，将输出全部记录",
            file=sys.stderr,
        )

    banner(2, 3, f"随机抽取 {actual_target:,} 条训练数据")
    suffix = ".jsonl.gz" if output_name.endswith(".jsonl.gz") else ".jsonl"
    temporary = tempfile.NamedTemporaryFile(
        prefix=".recipe_sample_",
        suffix=suffix,
        dir=output_dir,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        selected, sha256 = sample_records(
            source_path,
            temporary_path,
            population,
            requested,
            args.seed,
            args.progress_every,
        )
        source_after_sample = source_path.stat()
        if (source_before.st_size, source_before.st_mtime_ns) != (
            source_after_sample.st_size,
            source_after_sample.st_mtime_ns,
        ):
            raise RuntimeError("源文件在采样期间发生了变化，请停止写入后重试")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    banner(3, 3, "写入采样报告")
    report = {
        "pipeline_version": __version__,
        "source": str(source_path),
        "output": str(output_path),
        "source_total_lines": total_lines,
        "source_nonempty_records": population,
        "source_blank_lines": blank_lines,
        "requested_sample_size": requested,
        "selected_records": selected,
        "seed": args.seed,
        "algorithm": "two_pass_sequential_uniform_selection",
        "memory_behavior": "constant_memory",
        "output_order": "source_order_of_randomly_selected_records",
        "selected_records_validation": "instruction/input/output fields are strings",
        "content_sha256": sha256,
    }
    report_temporary = report_path.with_name(f".{report_path.name}.tmp")
    try:
        with report_temporary.open("w", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
            target.write("\n")
        os.replace(report_temporary, report_path)
    finally:
        report_temporary.unlink(missing_ok=True)

    print(
        f"采样完成\n"
        f"源数据：{population:,} 条\n"
        f"抽取数据：{selected:,} 条\n"
        f"随机种子：{args.seed}\n"
        f"样本文件：{output_path}\n"
        f"采样报告：{report_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
