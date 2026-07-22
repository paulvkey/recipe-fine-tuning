#!/usr/bin/env python3
"""Build high-confidence typo rules and protected words from a corpus sample."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import IO, Any

from recipe_pipeline import __version__
from recipe_pipeline.quality import load_protected_words, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT_ROOT / "data/base"
UNKNOWN = {"", "unknown", "none", "null", "未知", "无"}
INGREDIENT_PREFIXES = (
    "适量", "少量", "少许", "一点点", "一些", "若干", "大约", "约", "半根", "半个", "半颗", "一小把",
    "一大把", "一小块", "一小撮", "一碗", "一勺", "一匙", "半勺", "半汤匙",
    "大匙", "小匙", "汤匙", "茶匙", "大勺", "小勺", "瓶盖", "瓶", "罐",
    "克", "千克", "公斤", "斤", "两",
    "毫升", "升", "个", "只", "片", "根", "颗", "粒", "块", "杯", "勺",
    "瓣", "条", "朵", "头", "枚", "份", "盒", "把",
)
INGREDIENT_NOISE = ("可不", "根据", "喜欢", "需要", "材料", "做法", "备用", "切成", "加入")
NON_INGREDIENT_TERMS = {
    "包装", "高压锅", "面包机", "烤箱", "主锅", "锅", "模具", "容器", "工具",
    "保鲜膜", "锡纸", "烤盘", "油纸", "厨房纸", "步骤", "图片", "视频",
}
GENERIC_PROTECTED_TERMS = {
    "适量", "少量", "少许", "若干", "多一些", "看量", "据个人量", "自己看着办",
    "据食量口味自定义", "用量勺", "小料", "材料", "馅儿料", "镜面用",
}


def open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从原始食谱的随机样本生成纠错规则和保护词。")
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--min-typo-count", type=int, default=3)
    parser.add_argument("--min-protected-count", type=int, default=10)
    parser.add_argument(
        "--min-ingredient-protected-count",
        type=int,
        default=30,
        help="食材保护词最低样本频次，默认 30；菜名仍使用 --min-protected-count",
    )
    parser.add_argument("--scan-limit", type=int, help="测试用：最多扫描 N 个非空行")
    parser.add_argument("--base-typo-rules", type=Path, default=BASE_DATA / "typo_rules.json")
    parser.add_argument("--base-protected-words", type=Path, default=BASE_DATA / "protected_words.txt")
    parser.add_argument("--canonical-terms", type=Path, default=BASE_DATA / "canonical_terms.json")
    parser.add_argument("--confusion-map", type=Path, default=BASE_DATA / "confusion_map.json")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    if (
        args.sample_size <= 0
        or args.min_typo_count <= 0
        or args.min_protected_count <= 0
        or args.min_ingredient_protected_count <= 0
    ):
        parser.error("采样数量和频率阈值必须大于 0")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def project_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    projected: dict[str, Any] = {}
    for field in ("name", "dish", "description"):
        projected[field] = normalize_text(record.get(field), preserve_newlines=True)
    for field in ("recipeIngredient", "recipeInstructions"):
        value = record.get(field, [])
        if isinstance(value, str):
            value = [value]
        items: list[str] = []
        if isinstance(value, list):
            for item in value:
                text = normalize_text(item, preserve_newlines=True)
                if text:
                    items.append(text)
        projected[field] = items
    return projected


def reservoir_sample(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(args.seed)
    sample: list[dict[str, Any]] = []
    stats = Counter()
    with open_text(args.input) as source:
        for line in source:
            if not line.strip():
                stats["blank"] += 1
                continue
            stats["scanned"] += 1
            try:
                projected = project_record(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                projected = None
                stats["invalid"] += 1
            if projected is not None:
                stats["valid"] += 1
                if len(sample) < args.sample_size:
                    sample.append(projected)
                else:
                    index = rng.randrange(stats["valid"])
                    if index < args.sample_size:
                        sample[index] = projected
            if args.progress_every and stats["scanned"] % args.progress_every == 0:
                print(f"  [建库进度] 已扫描 {stats['scanned']:,} 行", file=sys.stderr)
            if args.scan_limit and stats["scanned"] >= args.scan_limit:
                break
    stats["sampled"] = len(sample)
    return sample, dict(stats)


def field_texts(record: dict[str, Any], field: str) -> list[str]:
    value = record.get(field, "")
    return value if isinstance(value, list) else [value]


def possible_variants(term: str, confusion: dict[str, list[str]]) -> set[str]:
    variants: set[str] = set()
    for index, char in enumerate(term):
        for wrong_char in confusion.get(char, []):
            variants.add(term[:index] + wrong_char + term[index + 1 :])
    return variants


def mine_typo_rules(
    sample: list[dict[str, Any]],
    canonical_document: dict[str, Any],
    confusion_document: dict[str, Any],
    base_document: dict[str, Any],
    minimum: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    confusion = confusion_document["characters"]
    existing = {
        rule["pattern"]
        for rule in base_document.get("rules", [])
        if not rule.get("regex")
    }
    found: list[dict[str, Any]] = []
    variant_terms: set[str] = set()
    auto_rules: list[dict[str, Any]] = []
    for specification in canonical_document["terms"]:
        correct = specification["term"]
        fields = specification["fields"]
        for variant in sorted(possible_variants(correct, confusion)):
            count = sum(
                text.count(variant)
                for record in sample
                for field in fields
                for text in field_texts(record, field)
            )
            if not count:
                continue
            variant_terms.add(variant)
            already_exists = variant in existing
            candidate = {
                "variant": variant,
                "replacement": correct,
                "fields": fields,
                "sample_count": count,
                "already_in_base": already_exists,
                "promoted": count >= minimum and not already_exists,
            }
            found.append(candidate)
            if candidate["promoted"]:
                digest = hashlib.blake2b(f"{variant}>{correct}".encode(), digest_size=4).hexdigest()
                auto_rules.append(
                    {
                        "id": f"auto_{digest}",
                        "enabled": True,
                        "fields": fields,
                        "pattern": variant,
                        "replacement": correct,
                        "reason": f"随机样本中出现 {count} 次；由标准烹饪词和易混字映射生成",
                        "source": "sample_mining",
                    }
                )
    merged = {
        "version": 1,
        "description": "基础规则与本次语料采样自动生成规则的合并结果。",
        "rules": base_document.get("rules", []) + auto_rules,
    }
    return merged, found, variant_terms


def ingredient_terms(text: str) -> set[str]:
    if re.fullmatch(r"【[^】]+】", text.strip()):
        return set()
    terms: set[str] = set()
    for term in re.findall(r"[\u4e00-\u9fff]{2,12}", text):
        term = re.sub(
            r"^[一二两三四五六七八九十半几]+(?:小|大|整)?(?:个|只|片|根|颗|粒|块|杯|碗|勺|匙|汤匙|茶匙|撮|瓣|条|朵|头|枚|份|盒|把)",
            "",
            term,
        )
        term = re.sub(
            r"(?:半?[个只片根颗粒块杯勺匙瓣条朵头枚份盒把])$",
            "",
            term,
        )
        term = re.sub(r"(?:适量|少量|少许|若干|备用|可选|装饰用)$", "", term)
        changed = True
        while changed:
            changed = False
            for prefix in INGREDIENT_PREFIXES:
                if term.startswith(prefix) and len(term) > len(prefix):
                    term = term[len(prefix) :]
                    changed = True
                    break
        if (
            2 <= len(term) <= 6
            and term not in GENERIC_PROTECTED_TERMS
            and term not in NON_INGREDIENT_TERMS
            and not term.endswith("用")
            and not any(noise in term for noise in INGREDIENT_NOISE)
        ):
            terms.add(term)
    return terms


def mine_protected_words(
    sample: list[dict[str, Any]],
    base_words: list[str],
    excluded_variants: set[str],
    dish_minimum: int,
    ingredient_minimum: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    source_counts: dict[str, Counter[str]] = {
        "dish": Counter(),
        "ingredient": Counter(),
    }
    sources: dict[str, set[str]] = {}
    for record in sample:
        dish = normalize_text(record.get("dish"))
        if dish.casefold() not in UNKNOWN and 2 <= len(dish) <= 12:
            source_counts["dish"][dish] += 1
            sources.setdefault(dish, set()).add("dish")
        for ingredient in record.get("recipeIngredient", []):
            for term in ingredient_terms(ingredient):
                source_counts["ingredient"][term] += 1
                sources.setdefault(term, set()).add("ingredient")

    base_set = set(base_words)
    all_terms = set(source_counts["dish"]) | set(source_counts["ingredient"])
    ranked_terms = sorted(
        all_terms,
        key=lambda term: (
            source_counts["dish"][term] + source_counts["ingredient"][term],
            term,
        ),
        reverse=True,
    )
    discovered = [
        {
            "term": term,
            "sample_count": source_counts["dish"][term] + source_counts["ingredient"][term],
            "source_counts": {
                source: source_counts[source][term]
                for source in ("dish", "ingredient")
                if source_counts[source][term]
            },
            "sources": sorted(sources[term]),
        }
        for term in ranked_terms
        if (
            source_counts["dish"][term] >= dish_minimum
            or source_counts["ingredient"][term] >= ingredient_minimum
        )
        and term not in base_set
        and not any(variant in term for variant in excluded_variants)
    ]
    merged = list(dict.fromkeys(base_words + [item["term"] for item in discovered]))
    return merged, discovered


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  [建库 1/3] 扫描原始数据并随机采样最多 {args.sample_size:,} 条",
        file=sys.stderr,
    )
    sample, scan_stats = reservoir_sample(args)
    print(
        f"  [建库 2/3] 分析 {len(sample):,} 条样本中的错字候选和高频保护词",
        file=sys.stderr,
    )
    base_rules = load_json(args.base_typo_rules)
    merged_rules, typo_candidates, variant_terms = mine_typo_rules(
        sample,
        load_json(args.canonical_terms),
        load_json(args.confusion_map),
        base_rules,
        args.min_typo_count,
    )
    protected_words, protected_candidates = mine_protected_words(
        sample,
        load_protected_words(args.base_protected_words),
        variant_terms,
        args.min_protected_count,
        args.min_ingredient_protected_count,
    )

    print("  [建库 3/3] 合并基础配置并写入自动生成配置", file=sys.stderr)
    rules_path = args.output_dir / "typo_rules.generated.json"
    words_path = args.output_dir / "protected_words.generated.txt"
    report_path = args.output_dir / "bootstrap_report.json"
    with rules_path.open("w", encoding="utf-8") as target:
        json.dump(merged_rules, target, ensure_ascii=False, indent=2)
        target.write("\n")
    with words_path.open("w", encoding="utf-8") as target:
        target.write("# 基础保护词与本次随机样本高频词的合并结果。\n")
        target.write("\n".join(protected_words) + "\n")
    report = {
        "pipeline_version": __version__,
        "input": str(args.input),
        "scan": scan_stats,
        "parameters": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "min_typo_count": args.min_typo_count,
            "min_protected_count": args.min_protected_count,
            "min_ingredient_protected_count": args.min_ingredient_protected_count,
        },
        "typo_candidates": typo_candidates,
        "generated_typo_rules": sum(1 for item in typo_candidates if item["promoted"]),
        "generated_protected_words": len(protected_candidates),
        "protected_word_candidates": protected_candidates,
        "outputs": {"typo_rules": str(rules_path), "protected_words": str(words_path)},
    }
    with report_path.open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(
        f"采样建库完成：规则 {report['generated_typo_rules']} 条，保护词 {len(protected_candidates)} 条",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
