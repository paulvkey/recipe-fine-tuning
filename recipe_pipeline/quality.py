"""Shared, dependency-free recipe cleaning and quality checks."""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TEXT_FIELDS = ("name", "dish", "description")
LIST_FIELDS = ("recipeIngredient", "recipeInstructions", "keywords")
UNKNOWN_VALUES = {"", "unknown", "none", "null", "未知", "无"}
MAX_OUTPUT_CHARS = 8_000
MIN_OUTPUT_CHARS = 35
MEDIA_RULE_IDS = {"trailing_media_reference", "standalone_media_step"}
LONG_PROCESS_WORDS = re.compile(
    r"腌|盐渍|浸泡|泡发|发酵|醒发|醒面|风干|晾|晒|酿|保存|存放|密封|冷藏|冷冻|保质|赏味|内吃完|咸蛋|泡菜|腊"
)
LONG_PROCESS_TITLE_WORDS = re.compile(r"腌|盐渍|发酵|风干|酿|酒|咸蛋|泡菜|腊|果脯|蜜饯")
GENERIC_TITLE = re.compile(
    r"^(?:(?:传图|上传)(?:专用)?|杂七杂八(?:的)?记录|食材之美|"
    r"(?:我的)?(?:作品|成品)(?:记录|集|合集)|(?:早餐|便当|蛋糕|烘焙|减脂餐)?打卡(?:记录)?|"
    r"记录(?:我做的)?(?:家常菜|美食|早餐|便当)?)$",
    re.I,
)
INGREDIENT_PLACEHOLDERS = {
    "材料", "配料", "食材", "用料", "用量", "材料如下", "配料如下", "食材如下", "见步骤",
}
ACTION_WORDS = re.compile(
    r"切|洗|煮|炒|煎|炸|蒸|烤|炖|焖|拌|加|放|倒|搅|揉|腌|泡|发酵|打发|打碎|磨|压|擀|包|卷|撒|淋|铺|抹|涂|摆|塞|挤|蘸|兑|冲|调味|装盘|取出|捞|沥|冷藏|冷冻|冰箱|融化|预热|烧开|熬|焯|汆|烫|过滤|混合|去皮|去核|下锅"
)
PROMPT_TEMPLATES = (
    "{name}怎么做？",
    "在家怎么做{name}？",
    "做{name}需要哪些食材和步骤？",
    "{name}的家常做法是什么？",
    "能告诉我{name}怎么做吗？",
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    fields: frozenset[str]
    pattern: re.Pattern[str]
    replacement: str
    reason: str
    literal: bool


class ProtectedWords(list[str]):
    """List-compatible protected terms with a lazy typo-fragment index."""

    def __init__(self, words: list[str]) -> None:
        super().__init__(words)
        self._fragment_cache: dict[str, list[str]] = {}

    def containing(self, fragment: str) -> list[str]:
        if fragment not in self._fragment_cache:
            self._fragment_cache[fragment] = [word for word in self if fragment in word]
        return self._fragment_cache[fragment]


def load_rules(path: Path) -> list[Rule]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    rules: list[Rule] = []
    for item in document.get("rules", []):
        if not item.get("enabled", True):
            continue
        literal = not item.get("regex", False)
        expression = re.escape(item["pattern"]) if literal else item["pattern"]
        rules.append(
            Rule(
                rule_id=item["id"],
                fields=frozenset(item["fields"]),
                pattern=re.compile(expression),
                replacement=item.get("replacement", ""),
                reason=item.get("reason", ""),
                literal=literal,
            )
        )
    # Specific/long expressions run before broad expressions.
    return sorted(rules, key=lambda rule: len(rule.pattern.pattern), reverse=True)


def load_protected_words(path: Path) -> ProtectedWords:
    with path.open(encoding="utf-8") as source:
        return ProtectedWords(
            [line.strip() for line in source if line.strip() and not line.lstrip().startswith("#")]
        )


def normalize_text(value: Any, preserve_newlines: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    # NFC composes equivalent Unicode sequences without changing Chinese
    # punctuation, full-width symbols or kaomoji as NFKC would.
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(
        char
        for char in value
        if char in "\n\t"
        or (
            unicodedata.category(char) != "Cc"
            and (unicodedata.category(char) != "Cf" or char == "\u200d")
        )
    )
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    return re.sub(r"\s+", " ", value).strip()


def protected_spans(text: str, words: Iterable[str], fragment: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    candidates = words.containing(fragment) if isinstance(words, ProtectedWords) else words
    for word in candidates:
        start = 0
        while (index := text.find(word, start)) >= 0:
            spans.append((index, index + len(word)))
            start = index + len(word)
    return spans


def apply_rules(
    text: str,
    field: str,
    rules: list[Rule],
    protected_words: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    for rule in rules:
        if field not in rule.fields:
            continue
        count = 0
        span_cache: dict[str, list[tuple[int, int]]] = {}

        def replace(match: re.Match[str]) -> str:
            nonlocal count
            if rule.literal:
                fragment = match.group(0)
                spans = span_cache.setdefault(
                    fragment, protected_spans(text, protected_words, fragment)
                )
                if any(match.start() < end and match.end() > start for start, end in spans):
                    return fragment
            count += 1
            return match.expand(rule.replacement)

        text = rule.pattern.sub(replace, text)
        if count:
            changes.append(
                {"rule": rule.rule_id, "field": field, "count": count, "reason": rule.reason}
            )
    return text, changes


def repair_ingredient_order(text: str) -> tuple[str, bool]:
    """Repair common scraper output such as ``克面粉100`` -> ``100克面粉``."""
    original = text
    reversed_quantity = re.fullmatch(
        r"(?P<unit>千克|公斤|毫升|克|升|勺|匙|个|只|片|根|颗|粒|块|杯|瓣|条|朵|头|枚|份|盒|把)"
        r"(?P<name>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z（）()、/\s]*?)"
        r"[。.:：\s]*(?P<amount>\d+(?:\.\d+)?|半|一|二|两|三|四|五|六|七|八|九|十)",
        text,
    )
    if reversed_quantity:
        text = (
            reversed_quantity.group("amount")
            + reversed_quantity.group("unit")
            + reversed_quantity.group("name").strip()
        )
    text = re.sub(r"^适量克(?=[\u4e00-\u9fff])", "适量", text)
    text = re.sub(
        r"^(半|一|二|两|三|四|五|六|七|八|九|十)(根|个|颗|片|块|只|瓣|条)\2",
        r"\1\2",
        text,
    )
    return text, text != original


def is_generic_title(text: str) -> bool:
    compact = re.sub(r"[\s_\-—:：|｜/\\]+", "", text).strip("。.!！~～")
    return bool(GENERIC_TITLE.fullmatch(compact))


def clean_record(
    record: Any,
    typo_rules: list[Rule],
    noise_rules: list[Rule],
    protected_words: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(record, dict):
        return None, [], [{"code": "not_object", "detail": "JSON 行不是对象"}]

    cleaned: dict[str, Any] = {}
    changes: list[dict[str, Any]] = []
    flags: list[dict[str, str]] = []

    for field in TEXT_FIELDS:
        text = normalize_text(record.get(field), preserve_newlines=field == "description")
        text, field_changes = apply_rules(text, field, typo_rules, protected_words)
        text, noise_changes = apply_rules(text, field, noise_rules, protected_words)
        cleaned[field] = normalize_text(text, preserve_newlines=field == "description")
        changes.extend(field_changes + noise_changes)

    for field in LIST_FIELDS:
        raw_items = record.get(field, [])
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raw_items = []
        items: list[str] = []
        removed_media_steps = 0
        for index, raw_item in enumerate(raw_items):
            text = normalize_text(raw_item, preserve_newlines=True)
            text, field_changes = apply_rules(text, field, typo_rules, protected_words)
            before_noise = text
            strong_media_dependency = bool(
                field in {"recipeIngredient", "recipeInstructions"}
                and re.search(
                    r"(?:(?:材料|配料|食材|用料|用量|比例|调法|做法|步骤).{0,6}"
                    r"(?:看|见|参考)(?:图|图片|视频)|"
                    r"(?:直接|只能|只好|就|只|主要|具体|详细)\s*"
                    r"(?:看|见|参考)(?:图|图片|视频)(?:做|操作|理解|意会|吧)?|"
                    r"不知道怎么(?:描述|说).{0,10}(?:看|见|参考)(?:图|图片|视频))",
                    before_noise,
                )
            )
            text, noise_changes = apply_rules(text, field, noise_rules, protected_words)
            text = normalize_text(text, preserve_newlines=True)
            if field == "recipeIngredient":
                text, repaired = repair_ingredient_order(text)
                if repaired:
                    changes.append(
                        {
                            "rule": "repair_ingredient_order",
                            "field": field,
                            "count": 1,
                            "reason": "修复抓取数据中数量、单位和食材顺序颠倒",
                        }
                    )
                compact_ingredient = re.sub(r"[\s:：,，。；;（）()]+", "", text)
                if compact_ingredient in INGREDIENT_PLACEHOLDERS:
                    text = ""
                    changes.append(
                        {
                            "rule": "remove_ingredient_placeholder",
                            "field": field,
                            "count": 1,
                            "reason": "删除不包含实际食材的材料或配料占位项",
                        }
                    )
            changes.extend(field_changes + noise_changes)
            if strong_media_dependency:
                flags.append(
                    {
                        "code": "media_reference",
                        "field": field,
                        "detail": "关键食材、用量或操作依赖缺失的图片或视频",
                    }
                )
            if text:
                items.append(text)
            elif before_noise and field == "recipeInstructions":
                if any(change["rule"] in MEDIA_RULE_IDS for change in noise_changes):
                    removed_media_steps += 1
        cleaned[field] = items
        if field == "recipeInstructions" and raw_items and items:
            # One discarded image-only step among two can still leave a usable
            # recipe. Quarantine only when at least two and half of all steps
            # contained no text-independent instruction.
            if removed_media_steps >= 2 and removed_media_steps / len(raw_items) >= 0.5:
                flags.append(
                    {
                        "code": "media_dependent",
                        "detail": f"{removed_media_steps}/{len(raw_items)} 个步骤完全依赖缺失图片",
                    }
                )

    name = cleaned["name"]
    dish = cleaned["dish"]
    if name.casefold() in UNKNOWN_VALUES:
        cleaned["name"] = dish
    elif (
        len(name) > 60
        and dish.casefold() not in UNKNOWN_VALUES
        and not is_generic_title(dish)
        and len(dish) <= 30
    ):
        cleaned["name"] = dish
        changes.append(
            {
                "rule": "short_dish_name_fallback",
                "field": "name",
                "count": 1,
                "reason": "食谱标题过长，使用简短 dish 生成自然提示词",
            }
        )
    elif (
        is_generic_title(name)
        and dish.casefold() not in UNKNOWN_VALUES
        and not is_generic_title(dish)
        and dish != name
        and len(dish) <= 30
    ):
        cleaned["name"] = dish
        changes.append(
            {
                "rule": "generic_title_dish_fallback",
                "field": "name",
                "count": 1,
                "reason": "标题是传图、记录或打卡占位词，改用有效 dish",
            }
        )
    if cleaned["name"].casefold() in UNKNOWN_VALUES:
        flags.append({"code": "missing_name", "detail": "name 和 dish 均无有效菜名"})
    if not cleaned["recipeIngredient"]:
        flags.append({"code": "missing_ingredients", "detail": "没有有效食材"})
    if not cleaned["recipeInstructions"]:
        flags.append({"code": "missing_instructions", "detail": "没有有效制作步骤"})
    return cleaned, changes, flags


def detect_anomalies(record: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if is_generic_title(record.get("name", "")):
        flags.append(
            {
                "code": "generic_title",
                "field": "name",
                "detail": "菜名是传图、记录、作品集或打卡类占位标题",
            }
        )
    field_values: list[tuple[str, str]] = []
    for field in TEXT_FIELDS:
        field_values.append((field, record.get(field, "")))
    for field in ("recipeIngredient", "recipeInstructions"):
        field_values.extend((field, item) for item in record.get(field, []))

    explicit_temp = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|摄氏度)", re.I)
    ambiguous_temp = re.compile(r"(\d{3,4}(?:\.\d+)?)\s*度")
    duration = re.compile(r"(\d+(?:\.\d+)?)\s*(分钟|小时|天)")
    huge_amount = re.compile(r"(\d{5,}(?:\.\d+)?)\s*(?:克|g|毫升|ml)(?![A-Za-z])", re.I)

    seen: set[tuple[str, str]] = set()
    for field, text in field_values:
        if not text:
            continue
        candidates: list[tuple[str, str]] = []
        if "�" in text:
            candidates.append(("mojibake", "包含乱码替换字符"))
        if re.search(
            r"(?:微信(?:公众号)?|公众号|个人微信|新浪微博|微博|vx|v信)\s*[:：]?\s*[@A-Za-z0-9_-]{5,}",
            text,
            re.I,
        ):
            candidates.append(("contact_info", "疑似包含联系方式"))
        if re.search(r"https?://|www\.", text, re.I):
            candidates.append(("external_link", "清理后仍包含外部链接"))
        if field == "recipeInstructions" and re.search(
            r"(?:(?:请|具体|详情|主要).{0,6}(?:看|见|参考).{0,4}(?:图|图片|视频)|"
            r"(?:做法|步骤|调法).{0,6}(?:看|见|参考).{0,4}(?:图|图片|视频))",
            text,
        ):
            candidates.append(("media_reference", "关键操作仍依赖缺失的图片或视频"))
        if field == "recipeIngredient" and re.search(
            r"(?:材料|配料|食材|用料|用量|比例).{0,4}(?:看|见|参考)(?:图|图片|视频)",
            text,
        ):
            candidates.append(("media_reference", "食材或用量仍依赖缺失的图片或视频"))
        if field in {"description", "recipeInstructions"}:
            for match in explicit_temp.finditer(text):
                value = float(match.group(1))
                # Negative temperatures can be valid for freezing; only impossible
                # high Celsius values are held for review.
                if value > 350:
                    candidates.append(("temperature", f"可疑摄氏温度：{match.group(0)}"))
            for match in ambiguous_temp.finditer(text):
                if float(match.group(1)) > 500:
                    candidates.append(("temperature", f"可疑温度：{match.group(0)}"))
        if field == "recipeInstructions":
            for match in duration.finditer(text):
                value, unit = float(match.group(1)), match.group(2)
                local_context = text[max(0, match.start() - 30) : match.end() + 30]
                animal_age = unit == "天" and bool(
                    re.search(
                        rf"{re.escape(match.group(0))}(?:以上|以下|左右)?(?:大的|的)?\s*(?:仔)?(?:鸡|鸭|鹅|鸽)",
                        local_context,
                    )
                )
                if animal_age:
                    continue
                long_process = bool(
                    LONG_PROCESS_WORDS.search(local_context)
                    or LONG_PROCESS_TITLE_WORDS.search(record.get("name", ""))
                )
                ordinary_limit = {"分钟": 1440, "小时": 168, "天": 30}[unit]
                long_process_limit = {"分钟": 10080, "小时": 2160, "天": 730}[unit]
                limit = long_process_limit if long_process else ordinary_limit
                if value > limit:
                    candidates.append(("duration", f"可疑时长：{match.group(0)}"))
        if field == "recipeIngredient":
            for match in huge_amount.finditer(text):
                candidates.append(("amount", f"可疑用量：{match.group(0)}"))
        for code, detail in candidates:
            key = (code, detail)
            if key not in seen:
                seen.add(key)
                flags.append({"code": code, "field": field, "detail": detail})

    output_length = len(format_output(record))
    if output_length > MAX_OUTPUT_CHARS:
        flags.append(
            {
                "code": "overlong_recipe",
                "field": "output",
                "detail": f"输出长度 {output_length} 字符，超过 {MAX_OUTPUT_CHARS}",
            }
        )

    meaningful_ingredients = "".join(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", "".join(record["recipeIngredient"]))
    )
    meaningful_steps = "".join(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", "".join(record["recipeInstructions"]))
    )
    ingredient_placeholder = re.compile(
        r"^[\W_]*(?:看图|见图|如图|适量|少许|若干|随意|任意|爱心|耐心|爱)[\W_]*$"
    )
    step_placeholder = re.compile(
        r"^[\W_]*(?:\d+|看图(?:一|二|三|\d+)?|见图(?:一|二|三|\d+)?|如图|图片整理|"
        r"传图专用|记录(?:我做的)?(?:家常菜|美食)?|求关注|待更新|下次补上|"
        r"步骤忘记(?:拍|写)(?:了)?(?:下次补上)?|忘记(?:拍|写)步骤(?:了)?|"
        r"看视频|见视频|见上文|见文字|详细步骤和用料看下面|图片都有详细说法(?:啦)?|"
        r"好吃|美味|完成|成功|可爱|一次成功|开吃)[\W_]*$",
        re.I,
    )
    useful_steps = [item for item in record["recipeInstructions"] if not step_placeholder.fullmatch(item)]
    all_ingredients_placeholder = all(
        ingredient_placeholder.fullmatch(item) for item in record["recipeIngredient"]
    )
    normalized_steps = [
        re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", item).casefold() for item in useful_steps
    ]
    repeated_steps = bool(
        len(normalized_steps) >= 4
        and len(set(normalized_steps)) <= max(1, len(normalized_steps) // 4)
    )
    has_action = any(ACTION_WORDS.search(item) for item in useful_steps) or any(
        re.search(r"(?:°|℃|度).{0,8}\d+(?:\.\d+)?\s*(?:分钟|小时)", item)
        for item in useful_steps
    )
    low_information_reasons: list[str] = []
    if output_length < MIN_OUTPUT_CHARS:
        low_information_reasons.append("输出过短")
    if len(meaningful_ingredients) < 2 and len(meaningful_steps) < 20:
        low_information_reasons.append("食材和步骤信息均不足")
    if len(meaningful_steps) < 8:
        low_information_reasons.append("步骤有效字符过少")
    if all_ingredients_placeholder:
        low_information_reasons.append("食材只有占位内容")
    if not useful_steps:
        low_information_reasons.append("没有可执行步骤")
    if repeated_steps:
        low_information_reasons.append("步骤高度重复")
    if not has_action and sum(len(item) for item in useful_steps) < 30:
        low_information_reasons.append("简短步骤缺少烹饪动作")
    if low_information_reasons:
        flags.append(
            {
                "code": "low_information",
                "field": "output",
                "detail": (
                    f"有效信息不足（{'、'.join(low_information_reasons)}）：输出 {output_length} 字符，"
                    f"食材有效字符 {len(meaningful_ingredients)}，步骤有效字符 {len(meaningful_steps)}"
                ),
            }
        )

    missing_quantity_prefix = re.compile(
        r"^(?:克|千克|公斤|毫升|升|勺|匙|个|只|片|根|颗|粒|块|杯|瓣|条|朵|头|枚|份|盒|把)(?=[\u4e00-\u9fffA-Za-z])"
    )
    malformed_count = sum(
        bool(missing_quantity_prefix.search(item)) for item in record["recipeIngredient"]
    )
    if malformed_count and malformed_count / len(record["recipeIngredient"]) >= 0.5:
        flags.append(
            {
                "code": "malformed_ingredients",
                "field": "recipeIngredient",
                "detail": f"{malformed_count}/{len(record['recipeIngredient'])} 条食材疑似缺少数量",
            }
        )
    return flags


def structural_failure(flags: list[dict[str, str]]) -> bool:
    codes = {flag["code"] for flag in flags}
    return bool(codes & {"not_object", "missing_name", "missing_ingredients", "missing_instructions"})


def choose_instruction(name: str) -> str:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=2).digest()
    template = PROMPT_TEMPLATES[int.from_bytes(digest, "big") % len(PROMPT_TEMPLATES)]
    return template.format(name=name)


def format_output(record: dict[str, Any]) -> str:
    answer: list[str] = []
    if record.get("description"):
        answer.append(f"简介：{record['description']}")
    answer.append("食材：\n" + "\n".join(f"- {item}" for item in record["recipeIngredient"]))
    answer.append(
        "制作步骤：\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(record["recipeInstructions"], 1))
    )
    return "\n\n".join(answer)


def format_training_record(record: dict[str, Any], instruction: str) -> dict[str, str]:
    return {"instruction": instruction, "input": "", "output": format_output(record)}
