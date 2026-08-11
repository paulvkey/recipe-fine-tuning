#!/usr/bin/env python3
"""Check that a local Hugging Face model snapshot is structurally complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 Qwen3-8B-Base 本地模型文件是否完整。")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    try:
        if not model_dir.is_dir():
            fail(f"模型目录不存在：{model_dir}")

        config_path = model_dir / "config.json"
        tokenizer_path = model_dir / "tokenizer.json"
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        for path in (config_path, tokenizer_path, tokenizer_config_path):
            if not path.is_file() or path.stat().st_size == 0:
                fail(f"缺少模型文件或文件为空：{path}")

        with config_path.open(encoding="utf-8") as source:
            config = json.load(source)
        if config.get("model_type") != "qwen3":
            fail(f"config.json 的 model_type 不是 qwen3：{config.get('model_type')!r}")

        index_path = model_dir / "model.safetensors.index.json"
        single_weight = model_dir / "model.safetensors"
        weight_files: set[Path] = set()
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as source:
                index = json.load(source)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                fail(f"权重索引缺少有效 weight_map：{index_path}")
            for filename in set(weight_map.values()):
                if not isinstance(filename, str) or Path(filename).name != filename:
                    fail(f"权重索引包含异常文件名：{filename!r}")
                weight_files.add(model_dir / filename)
        elif single_weight.is_file():
            weight_files.add(single_weight)
        else:
            fail("缺少 model.safetensors 或 model.safetensors.index.json")

        missing = [path for path in sorted(weight_files) if not path.is_file() or path.stat().st_size == 0]
        if missing:
            fail("缺少权重分片或文件为空：" + ", ".join(str(path) for path in missing))

        weight_bytes = sum(path.stat().st_size for path in weight_files)
        if weight_bytes < 15_000_000_000:
            fail(f"权重文件总大小异常，仅有 {weight_bytes / 1_000_000_000:.2f} GB")

        if not args.quiet:
            print("模型校验通过")
            print(f"目录：{model_dir}")
            print(f"模型类型：{config.get('model_type')}")
            print(f"权重分片：{len(weight_files)}")
            print(f"权重大小：{weight_bytes / 1_000_000_000:.2f} GB")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if not args.quiet:
            print(f"模型校验失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

