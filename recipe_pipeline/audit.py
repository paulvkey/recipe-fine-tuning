#!/usr/bin/env python3
"""First pass: audit a large recipe JSONL without modifying it."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO

from recipe_pipeline import __version__
from recipe_pipeline.quality import (
    clean_record,
    detect_anomalies,
    load_protected_words,
    load_rules,
    structural_failure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT_ROOT / "data/base"


def open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="流式审计食谱 JSONL，只生成报告，不修改原文件。")
    parser.add_argument("input", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--typo-rules", type=Path, default=BASE_DATA / "typo_rules.json")
    parser.add_argument("--noise-rules", type=Path, default=BASE_DATA / "noise_rules.json")
    parser.add_argument("--protected-words", type=Path, default=BASE_DATA / "protected_words.txt")
    parser.add_argument("--examples", type=int, default=5, help="每类问题最多保存的示例数")
    parser.add_argument("--limit", type=int, help="只审计前 N 个非空行")
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("  [审计 1/2] 流式扫描并统计结构、数值和上下文异常", file=sys.stderr)
    typo_rules = load_rules(args.typo_rules)
    noise_rules = load_rules(args.noise_rules)
    protected_words = load_protected_words(args.protected_words)
    counts: Counter[str] = Counter()
    issue_occurrences: Counter[str] = Counter()
    issue_records: Counter[str] = Counter()
    rule_applications: Counter[str] = Counter()
    rule_records: Counter[str] = Counter()
    rule_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)

    with open_text(args.input) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            counts["read"] += 1
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                counts["rejected"] += 1
                issue_occurrences["invalid_json"] += 1
                issue_records["invalid_json"] += 1
                if len(examples["invalid_json"]) < args.examples:
                    examples["invalid_json"].append(
                        {"line": line_number, "detail": str(error), "text": line[:300].rstrip()}
                    )
                continue

            cleaned, changes, flags = clean_record(raw, typo_rules, noise_rules, protected_words)
            if cleaned is not None:
                flags.extend(detect_anomalies(cleaned))
            changed_rule_ids: set[str] = set()
            for change in changes:
                rule_applications[change["rule"]] += int(change["count"])
                changed_rule_ids.add(change["rule"])
                if len(rule_examples[change["rule"]]) < args.examples:
                    rule_examples[change["rule"]].append(
                        {
                            "line": line_number,
                            "name": cleaned.get("name", "") if cleaned else "",
                            "field": change["field"],
                            "count": change["count"],
                        }
                    )
            for rule_id in changed_rule_ids:
                rule_records[rule_id] += 1
            issue_codes: set[str] = set()
            for flag in flags:
                code = flag["code"]
                issue_occurrences[code] += 1
                issue_codes.add(code)
                if len(examples[code]) < args.examples:
                    examples[code].append(
                        {
                            "line": line_number,
                            "name": cleaned.get("name", "") if cleaned else "",
                            "detail": flag.get("detail", ""),
                        }
                    )
            for code in issue_codes:
                issue_records[code] += 1
            if structural_failure(flags):
                counts["rejected"] += 1
            elif flags:
                counts["review"] += 1
            else:
                counts["clean"] += 1

            if args.progress_every and counts["read"] % args.progress_every == 0:
                print(f"  [审计进度] 已审计 {counts['read']:,} 行", file=sys.stderr)
            if args.limit and counts["read"] >= args.limit:
                break

    print("  [审计 2/2] 汇总问题计数、规则命中和示例", file=sys.stderr)
    report = {
        "pipeline_version": __version__,
        "input": str(args.input),
        "summary": dict(counts),
        "issue_counts": dict(issue_records.most_common()),
        "issue_record_counts": dict(issue_records.most_common()),
        "issue_occurrence_counts": dict(issue_occurrences.most_common()),
        "rule_record_counts": dict(rule_records.most_common()),
        "rule_application_counts": dict(rule_applications.most_common()),
        "counting_notes": {
            "record_counts": "每条食谱的同类问题或规则最多计数一次",
            "occurrence_or_application_counts": "同一食谱内的多次命中分别计数",
        },
        "rule_examples": dict(rule_examples),
        "examples": dict(examples),
        "rule_files": {
            "typo": str(args.typo_rules),
            "noise": str(args.noise_rules),
            "protected_words": str(args.protected_words),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"审计完成：{args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
