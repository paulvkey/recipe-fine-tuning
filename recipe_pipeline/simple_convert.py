#!/usr/bin/env python3
"""Legacy/simple JSONL converter without quality auditing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import IO, Any, Iterator


UNKNOWN_VALUES = {"", "unknown", "none", "null", "未知", "无"}
PROMPT_TEMPLATES = (
    "{name}怎么做？",
    "在家怎么做{name}？",
    "做{name}需要哪些食材和步骤？",
    "{name}的家常做法是什么？",
    "能告诉我{name}怎么做吗？",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐行转换食谱 JSONL，内存中始终只保留一条记录。"
    )
    parser.add_argument("input", type=Path, help="原始 JSONL 文件，也支持 .gz")
    parser.add_argument("output", type=Path, help="输出 JSONL 文件，也支持 .gz")
    parser.add_argument(
        "--errors",
        type=Path,
        help="可选：把无法解析/转换的原始行写入此文件",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只转换前 N 条有效样本，适合预览和测试",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="每处理多少行输出一次进度；设为 0 关闭",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if args.input.resolve() == args.output.resolve():
        parser.error("输入和输出不能是同一个文件")
    return args


def open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def clean_text(value: Any, *, preserve_newlines: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\u00a0", " ").strip()
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line)
    return re.sub(r"\s+", " ", value)


def clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = clean_text(item, preserve_newlines=True)
        if text:
            result.append(text)
    return result


def is_unknown(text: str) -> bool:
    return text.casefold() in UNKNOWN_VALUES


def choose_instruction(name: str) -> str:
    """Choose a deterministic, conversational prompt without multiplying samples."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=2).digest()
    index = int.from_bytes(digest, "big")
    template = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]
    return template.format(name=name)


def convert(record: Any) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None

    name = clean_text(record.get("name"))
    dish = clean_text(record.get("dish"))
    description = clean_text(record.get("description"), preserve_newlines=True)
    ingredients = clean_list(record.get("recipeIngredient"))
    instructions = clean_list(record.get("recipeInstructions"))
    if is_unknown(name):
        name = dish
    if is_unknown(name) or not ingredients or not instructions:
        return None
    answer_parts: list[str] = []
    if description:
        answer_parts.append(f"简介：{description}")
    answer_parts.append("食材：\n" + "\n".join(f"- {item}" for item in ingredients))
    answer_parts.append(
        "制作步骤：\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(instructions, 1))
    )

    return {
        "instruction": choose_instruction(name),
        "input": "",
        "output": "\n\n".join(answer_parts),
    }


def records(lines: Iterator[str]) -> Iterator[tuple[int, str]]:
    for line_number, line in enumerate(lines, 1):
        if line.strip():
            yield line_number, line


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)

    read_count = written_count = skipped_count = 0
    error_file = open_text(args.errors, "w") if args.errors else None
    try:
        with open_text(args.input, "r") as source, open_text(args.output, "w") as target:
            for line_number, line in records(source):
                read_count += 1
                try:
                    converted = convert(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    converted = None

                if converted is None:
                    skipped_count += 1
                    if error_file:
                        error_file.write(line)
                else:
                    target.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    written_count += 1

                if args.progress_every and read_count % args.progress_every == 0:
                    print(
                        f"已读取 {read_count:,} 行，写入 {written_count:,} 条，跳过 {skipped_count:,} 条",
                        file=sys.stderr,
                    )
                if args.limit is not None and written_count >= args.limit:
                    break
    finally:
        if error_file:
            error_file.close()

    print(
        f"完成：读取 {read_count:,} 行，写入 {written_count:,} 条，跳过 {skipped_count:,} 条 -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
