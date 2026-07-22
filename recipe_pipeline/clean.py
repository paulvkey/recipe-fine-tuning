#!/usr/bin/env python3
"""Clean, validate, split, deduplicate and convert recipe JSONL."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import IO, Any

from recipe_pipeline import __version__
from recipe_pipeline.quality import (
    MAX_OUTPUT_CHARS,
    MIN_OUTPUT_CHARS,
    choose_instruction,
    clean_record,
    detect_anomalies,
    format_training_record,
    load_protected_words,
    load_rules,
    structural_failure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT_ROOT / "data/base"


def open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def derived_path(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="流式清洗、质检、精确去重并转换食谱 JSONL。")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="干净训练集 JSONL")
    parser.add_argument("--review", type=Path, help="风险数据；默认在输出文件名后加 .review.jsonl")
    parser.add_argument("--rejected", type=Path, help="结构不合格数据；默认加 .rejected.jsonl")
    parser.add_argument("--report", type=Path, help="清洗统计；默认加 .report.json")
    parser.add_argument("--typo-rules", type=Path, default=BASE_DATA / "typo_rules.json")
    parser.add_argument("--noise-rules", type=Path, default=BASE_DATA / "noise_rules.json")
    parser.add_argument("--protected-words", type=Path, default=BASE_DATA / "protected_words.txt")
    parser.add_argument("--no-deduplicate", action="store_true", help="关闭基于菜名、食材和步骤的精确去重")
    parser.add_argument(
        "--target-count",
        type=int,
        help="从清洗、去重后的有效数据中随机选择 N 条；默认输出全部",
    )
    parser.add_argument("--selection-seed", type=int, default=20260722, help="随机选择种子")
    parser.add_argument("--scan-limit", type=int, help="测试用：最多处理 N 个非空原始记录")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    args.review = args.review or derived_path(args.output, ".review.jsonl")
    args.rejected = args.rejected or derived_path(args.output, ".rejected.jsonl")
    args.report = args.report or derived_path(args.output, ".report.json")
    resolved = [path.resolve() for path in (args.input, args.output, args.review, args.rejected, args.report)]
    if len(set(resolved)) != len(resolved):
        parser.error("输入、输出、审核、拒绝和报告文件必须互不相同")
    if args.target_count is not None and args.target_count <= 0:
        parser.error("--target-count 必须大于 0")
    if args.scan_limit is not None and args.scan_limit <= 0:
        parser.error("--scan-limit 必须大于 0")
    return args


def write_jsonl(target: IO[str], value: Any) -> None:
    target.write(json.dumps(value, ensure_ascii=False) + "\n")


def dedup_key(record: dict[str, Any]) -> str:
    content = [record["name"], record["recipeIngredient"], record["recipeInstructions"]]
    packed = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def select_random_records(
    spool_path: Path,
    output_path: Path,
    eligible_count: int,
    target_count: int,
    seed: int,
) -> int:
    """Select exactly N records uniformly with a sequential, constant-memory pass."""
    needed = min(target_count, eligible_count)
    remaining = eligible_count
    selected = 0
    rng = random.Random(seed)
    with spool_path.open("r", encoding="utf-8", newline="") as source, open_text(
        output_path, "w"
    ) as target:
        for line in source:
            if needed and rng.randrange(remaining) < needed:
                target.write(line)
                needed -= 1
                selected += 1
            remaining -= 1
    return selected


def main() -> int:
    args = parse_args()
    for path in (args.output, args.review, args.rejected, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    typo_rules = load_rules(args.typo_rules)
    noise_rules = load_rules(args.noise_rules)
    protected_words = load_protected_words(args.protected_words)
    stats: Counter[str] = Counter()
    issue_occurrences: Counter[str] = Counter()
    issue_records: Counter[str] = Counter()
    rule_applications: Counter[str] = Counter()
    rule_records: Counter[str] = Counter()

    print("  [清洗 1/3] 加载规则，准备流式清洗与质量检测", file=sys.stderr)

    db_file = tempfile.NamedTemporaryFile(prefix="recipe_dedup_", suffix=".sqlite3", delete=False)
    db_path = Path(db_file.name)
    db_file.close()
    database = sqlite3.connect(db_path)
    database.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY) WITHOUT ROWID")

    spool_path: Path | None = None
    if args.target_count is not None:
        spool_file = tempfile.NamedTemporaryFile(
            prefix="recipe_candidates_",
            suffix=".jsonl",
            dir=args.output.parent,
            delete=False,
        )
        spool_path = Path(spool_file.name)
        spool_file.close()
    candidate_path = spool_path or args.output

    try:
        with (
            open_text(args.input, "r") as source,
            open_text(candidate_path, "w") as candidate_target,
            open_text(args.review, "w") as review_target,
            open_text(args.rejected, "w") as rejected_target,
        ):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                if args.scan_limit and stats["read"] >= args.scan_limit:
                    break
                stats["read"] += 1
                try:
                    raw = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    stats["rejected"] += 1
                    issue_occurrences["invalid_json"] += 1
                    issue_records["invalid_json"] += 1
                    write_jsonl(
                        rejected_target,
                        {"line": line_number, "flags": [{"code": "invalid_json", "detail": str(error)}], "raw": line[:2000].rstrip()},
                    )
                    continue

                cleaned, changes, flags = clean_record(raw, typo_rules, noise_rules, protected_words)
                if cleaned is not None:
                    flags.extend(detect_anomalies(cleaned))
                changed_rule_ids: set[str] = set()
                for change in changes:
                    rule_applications[change["rule"]] += int(change["count"])
                    changed_rule_ids.add(change["rule"])
                for rule_id in changed_rule_ids:
                    rule_records[rule_id] += 1
                issue_codes: set[str] = set()
                for flag in flags:
                    issue_occurrences[flag["code"]] += 1
                    issue_codes.add(flag["code"])
                for code in issue_codes:
                    issue_records[code] += 1

                envelope = {"line": line_number, "flags": flags, "changes": changes, "record": cleaned}
                if structural_failure(flags) or cleaned is None:
                    stats["rejected"] += 1
                    write_jsonl(rejected_target, envelope)
                    continue
                candidate = format_training_record(cleaned, choose_instruction(cleaned["name"]))
                if flags:
                    stats["review"] += 1
                    envelope["candidate"] = candidate
                    write_jsonl(review_target, envelope)
                    continue

                if not args.no_deduplicate:
                    cursor = database.execute("INSERT OR IGNORE INTO seen VALUES (?)", (dedup_key(cleaned),))
                    if cursor.rowcount == 0:
                        stats["duplicates"] += 1
                        continue
                write_jsonl(candidate_target, candidate)
                stats["eligible"] += 1
                if args.target_count is None:
                    stats["clean"] += 1

                if stats["eligible"] % 10_000 == 0:
                    database.commit()
                if args.progress_every and stats["read"] % args.progress_every == 0:
                    print(
                        f"  [清洗进度] 读取 {stats['read']:,}，有效 {stats['eligible']:,}，审核 {stats['review']:,}，拒绝 {stats['rejected']:,}，重复 {stats['duplicates']:,}",
                        file=sys.stderr,
                    )
    finally:
        database.close()
        db_path.unlink(missing_ok=True)
        if sys.exc_info()[0] is not None and spool_path is not None:
            spool_path.unlink(missing_ok=True)

    try:
        if spool_path is not None:
            print(
                f"  [清洗 2/3] 从 {stats['eligible']:,} 条有效数据中随机选择最多 {args.target_count:,} 条",
                file=sys.stderr,
            )
            stats["clean"] = select_random_records(
                spool_path,
                args.output,
                stats["eligible"],
                args.target_count,
                args.selection_seed,
            )
            stats["not_selected"] = stats["eligible"] - stats["clean"]
            if stats["clean"] < args.target_count:
                print(
                    f"  [清洗提示] 仅有 {stats['eligible']:,} 条有效数据，少于请求的 {args.target_count:,} 条，已输出全部有效数据",
                    file=sys.stderr,
                )
        else:
            print("  [清洗 2/3] 未指定转换数量，保留全部有效数据", file=sys.stderr)
    finally:
        if spool_path is not None:
            spool_path.unlink(missing_ok=True)

    print("  [清洗 3/3] 写入清洗统计报告", file=sys.stderr)
    report = {
        "pipeline_version": __version__,
        "input": str(args.input),
        "output": str(args.output),
        "summary": dict(stats),
        # issue_counts remains as a compatibility alias, but now uses the more
        # useful affected-record interpretation instead of occurrence count.
        "issue_counts": dict(issue_records.most_common()),
        "issue_record_counts": dict(issue_records.most_common()),
        "issue_occurrence_counts": dict(issue_occurrences.most_common()),
        "rule_record_counts": dict(rule_records.most_common()),
        "rule_application_counts": dict(rule_applications.most_common()),
        "counting_notes": {
            "record_counts": "每条食谱的同类问题或规则最多计数一次",
            "occurrence_or_application_counts": "同一食谱内的多次命中分别计数",
        },
        "selection": {
            "requested": args.target_count,
            "selected": stats["clean"],
            "seed": args.selection_seed if args.target_count is not None else None,
            "stage": "after_cleaning_validation_and_deduplication",
        },
        "quality_policy": {
            "unicode_normalization": "NFC",
            "min_output_chars": MIN_OUTPUT_CHARS,
            "max_output_chars": MAX_OUTPUT_CHARS,
            "context_aware_duration_checks": True,
            "semantic_low_information_checks": True,
            "inline_media_deictic_cleanup": True,
            "ingredient_media_cleanup": True,
            "generic_title_checks": True,
            "animal_age_duration_exemption": True,
            "flagged_records_are_routed_to_review": True,
        },
        "rules": {
            "typo": str(args.typo_rules),
            "noise": str(args.noise_rules),
            "protected_words": str(args.protected_words),
        },
    }
    with args.report.open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"完成：{json.dumps(dict(stats), ensure_ascii=False)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
