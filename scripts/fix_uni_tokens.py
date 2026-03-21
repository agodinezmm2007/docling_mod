"""
Repair ligature artifacts in PagesJson item text.

This script rewrites selected PagesJson item text in-place. It does not modify
FullText or page aggregate content fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from wordfreq import zipf_frequency
from wordsegment import load as load_wordsegment
from wordsegment import segment as wordsegment_segment

try:
    from scripts.topic_text_reconstruction import classify_item
except ImportError:  # pragma: no cover
    from topic_text_reconstruction import classify_item  # type: ignore


load_wordsegment()


LIGATURE_UNI_MAP = {
    "FB00": "ff",
    "FB01": "fi",
    "FB02": "fl",
    "FB03": "ffi",
    "FB04": "ffl",
}

GLYPH_MAPPING_PATH = Path(__file__).with_name("glyph_mapping.json")
GLYPH_MAP = json.loads(GLYPH_MAPPING_PATH.read_text(encoding="utf-8"))
SAFE_DIRECT_GLYPHS = {token: entry["character"] for token, entry in GLYPH_MAP.items() if token not in {"GLYPH<11>", "GLYPH<14>"}}
GLYPH_LIGATURES = {
    "GLYPH<11>": "ff",
    "GLYPH<14>": "ffi",
}

UNI_TOKEN_RE = re.compile(r"/uni([0-9A-Fa-f]{4})", re.IGNORECASE)
UNI_LIGATURE_TOKEN_RE = re.compile(r"/uni(FB0[0-4])", re.IGNORECASE)
GLYPH_TOKEN_RE = re.compile(r"GLYPH<[^>]+>")
DOUBLE_SPACE_RE = re.compile(r"([A-Za-z]{2,})( {2,})([A-Za-z]{2,})")
WORD_RE = re.compile(r"[A-Za-z]{2,}")
URLISH_RE = re.compile(
    r"https?://|www\.|doi\b|pmid\b|arxiv\b|\.gov\b|\.edu\b|\.org\b|\.com\b|\.net\b",
    re.IGNORECASE,
)
AMBIGUOUS_MARGIN = 3.0
SEGMENT_BREAK_PENALTY = 0.75
LIGATURE_PARTS = ("fi", "fl", "ff", "ffi", "ffl")
URL_JOIN_CHARS = ".-_/?:#%=&"
DOUBLE_SPACE_SAFE_LABELS = {"text", "list_item", "section_header", "caption", "footnote", "page_footer"}
LIGATURE_SCOPE_CHOICES = ("topic", "non_reference", "all")


@dataclass
class Decision:
    replacement: str | None
    status: str
    reason: str
    join_score: float
    segmented_score: float
    segmented_tokens: list[str]


@dataclass
class LigatureSpan:
    span_start: int
    span_end: int
    original: str
    raw_letters: str
    token_context: str
    left: str
    right: str


def resolve_uni_tokens(text: str) -> str:
    """Replace /uniXXXX tokens with their decomposed text form."""
    if not isinstance(text, str) or "/uni" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        code = match.group(1).upper()
        if code in LIGATURE_UNI_MAP:
            return LIGATURE_UNI_MAP[code]
        return chr(int(code, 16))

    return UNI_TOKEN_RE.sub(repl, text)


def resolve_glyph_token(token: str) -> str | None:
    if token in SAFE_DIRECT_GLYPHS:
        return SAFE_DIRECT_GLYPHS[token]
    return None


def previous_nonspace_char(text: str, index: int) -> str | None:
    cursor = index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    if cursor < 0:
        return None
    return text[cursor]


def next_nonspace_char(text: str, index: int) -> str | None:
    cursor = index
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return None
    return text[cursor]


def is_degree_glyph_context(text: str, start: int, end: int) -> bool:
    left_window = text[max(0, start - 12):start]
    right_window = text[end:min(len(text), end + 12)]
    compact = re.sub(r"\s+", " ", f"{left_window}<G>{right_window}")

    if re.search(r"\d\s*<G>\s*[CFKcfk]\b", compact):
        return True
    if re.search(r"\(\s*<G>\s*[CFK]\b", compact):
        return True
    if re.search(r"\d\s*<G>\s*[×x]\s*\d", compact):
        return True
    if re.search(r"[NSWE]\s*\d+\s*<G>", compact, re.IGNORECASE):
        return True

    prev_char = previous_nonspace_char(text, start)
    next_char = next_nonspace_char(text, end)
    if prev_char and next_char and prev_char.isdigit() and next_char in "CFKcfk":
        return True
    if prev_char and next_char and prev_char.isdigit() and next_char.isdigit():
        return True
    return False


def build_ligature_span(text: str, start: int, end: int, ligature: str) -> LigatureSpan:
    left_scan = start
    while left_scan > 0 and text[left_scan - 1].isspace():
        left_scan -= 1
    left_end = left_scan
    while left_scan > 0 and text[left_scan - 1].isalpha():
        left_scan -= 1
    left = text[left_scan:left_end]
    span_start = left_scan if left else start

    right_scan = end
    while right_scan < len(text) and text[right_scan].isspace():
        right_scan += 1
    right_start = right_scan
    while right_scan < len(text) and text[right_scan].isalpha():
        right_scan += 1
    right = text[right_start:right_scan]
    span_end = right_scan if right else end
    if not right and right_start > end and right_start < len(text) and text[right_start] in URL_JOIN_CHARS:
        span_end = right_start

    original = text[span_start:span_end]
    raw_letters = f"{left}{ligature}{right}" if (left or right) else ligature
    token_start = span_start
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token_end = span_end
    while token_end < len(text) and not text[token_end].isspace():
        token_end += 1
    token_context = text[token_start:token_end]
    return LigatureSpan(span_start, span_end, original, raw_letters, token_context, left, right)


def project_segmentation(raw_letters: str, tokens: list[str]) -> str | None:
    """Map a lower-case segmentation back to the original casing."""
    if sum(len(token) for token in tokens) != len(raw_letters):
        return None

    offset = 0
    parts: list[str] = []
    for token in tokens:
        next_offset = offset + len(token)
        parts.append(raw_letters[offset:next_offset])
        offset = next_offset
    return " ".join(parts)


def is_urlish(text: str) -> bool:
    return bool(URLISH_RE.search(text))


def is_prose_like_label(label: str, *, table_cell: bool = False) -> bool:
    if table_cell:
        return True
    return label in DOUBLE_SPACE_SAFE_LABELS


def select_item_for_scope(item: dict[str, Any], page_no: int, scope: str) -> tuple[bool, str]:
    if scope == "topic":
        return classify_item(item, page_no)
    if scope == "non_reference":
        if item.get("is_reference", False):
            return False, "is_reference"
        return True, "non_reference"
    if scope == "all":
        return True, "all"
    raise ValueError(f"Unknown scope: {scope}")


class LexicalScorer:
    def __init__(self, corpus_counts: Counter[str] | None = None) -> None:
        self.corpus_counts = corpus_counts or Counter()

    def token_score(self, token: str) -> float:
        lowered = token.lower()
        score = zipf_frequency(lowered, "en")
        count = self.corpus_counts.get(lowered, 0)
        if count:
            score = max(score, 3.0 + min(math.log10(count + 1), 1.5))
        return score

    def phrase_score(self, tokens: list[str]) -> float:
        if not tokens:
            return float("-inf")
        return sum(self.token_score(token) for token in tokens) - SEGMENT_BREAK_PENALTY * (len(tokens) - 1)

    def segment(self, raw_letters: str) -> list[str]:
        lowered = raw_letters.lower()
        tokens = wordsegment_segment(lowered)
        if not tokens or "".join(tokens) != lowered or any(not token.isalpha() for token in tokens):
            return [lowered]
        return tokens

    def decide_uni(self, raw_letters: str, original: str, token_context: str) -> Decision:
        if is_urlish(token_context):
            join_score = self.token_score(raw_letters)
            return Decision(raw_letters, "auto", "urlish_join", join_score, join_score, [raw_letters.lower()])

        segmented_tokens = self.segment(raw_letters)
        join_score = self.token_score(raw_letters)
        segmented_score = self.phrase_score(segmented_tokens)

        if len(segmented_tokens) == 1:
            return Decision(raw_letters, "auto", "single_segment", join_score, segmented_score, segmented_tokens)

        projected = project_segmentation(raw_letters, segmented_tokens)
        if projected is None:
            return Decision(None, "review", "projection_failed", join_score, segmented_score, segmented_tokens)

        gap = segmented_score - join_score
        if gap >= AMBIGUOUS_MARGIN:
            return Decision(projected, "auto", "segmented_phrase", join_score, segmented_score, segmented_tokens)
        if gap <= 1.0:
            return Decision(raw_letters, "auto", "joined_phrase", join_score, segmented_score, segmented_tokens)

        return Decision(None, "review", "low_confidence", join_score, segmented_score, segmented_tokens)

    def decide_double_space(self, left: str, right: str, context_window: str) -> Decision:
        if is_urlish(context_window):
            return Decision(None, "skip", "urlish_context", 0.0, 0.0, [])

        joined = left + right
        segmented_tokens = self.segment(joined)
        join_score = self.token_score(joined)
        segmented_score = self.phrase_score(segmented_tokens)
        left_score = self.token_score(left)
        right_score = self.token_score(right)
        boundary_score = self.phrase_score([left.lower(), right.lower()])
        left_count = self.corpus_counts.get(left.lower(), 0)
        right_count = self.corpus_counts.get(right.lower(), 0)
        joined_count = self.corpus_counts.get(joined.lower(), 0)
        suspicious_split = (
            left.lower().endswith(LIGATURE_PARTS)
            or right.lower().startswith(LIGATURE_PARTS)
            or min(left_score, right_score) < 2.5
        )

        if len(segmented_tokens) == 1 and join_score >= boundary_score + 1.0:
            return Decision(joined, "auto", "double_space_join", join_score, boundary_score, segmented_tokens)
        if len(segmented_tokens) == 1 and joined_count >= max(10, left_count + right_count + 5):
            return Decision(joined, "auto", "double_space_join", join_score, boundary_score, segmented_tokens)
        if len(segmented_tokens) == 1 and join_score >= 3.0 and min(left_score, right_score) < 1.0:
            return Decision(joined, "auto", "double_space_join", join_score, boundary_score, segmented_tokens)
        if len(segmented_tokens) == 1 and join_score >= 3.5 and (left_count == 0 or right_count == 0):
            return Decision(joined, "auto", "double_space_join", join_score, boundary_score, segmented_tokens)
        if len(segmented_tokens) == 1 and join_score >= 3.5 and suspicious_split:
            return Decision(joined, "auto", "double_space_join", join_score, boundary_score, segmented_tokens)

        return Decision(None, "skip", "keep_split", join_score, segmented_score, segmented_tokens)


def build_corpus_counts(df: pd.DataFrame, scope: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in df.itertuples(index=False):
        pages_json = getattr(row, "PagesJson", "")
        if not isinstance(pages_json, str) or not pages_json or pages_json == "ANALYSIS_ERROR":
            continue

        try:
            pages = json.loads(pages_json)
        except (json.JSONDecodeError, TypeError):
            continue

        for page in pages:
            if not page:
                continue
            page_no = page.get("page_no", 0)
            for item in page.get("items", []):
                keep, _ = select_item_for_scope(item, page_no, scope)
                if not keep:
                    continue

                label = item.get("label", "")
                text = item.get("text", "")
                if isinstance(text, str) and is_prose_like_label(label):
                    if "/uni" not in text and "GLYPH<" not in text and not DOUBLE_SPACE_RE.search(text):
                        for word in WORD_RE.findall(text):
                            counts[word.lower()] += 1
                elif label == "table" and isinstance(text, list):
                    for row_dict in text:
                        if not isinstance(row_dict, dict):
                            continue
                        for value in row_dict.values():
                            if isinstance(value, str) and "/uni" not in value and "GLYPH<" not in value and not DOUBLE_SPACE_RE.search(value):
                                for word in WORD_RE.findall(value):
                                    counts[word.lower()] += 1
    return counts


def collect_review(
    reviews: list[dict[str, Any]] | None,
    context: dict[str, Any] | None,
    source: str,
    original: str,
    raw_letters: str,
    decision: Decision,
) -> None:
    if reviews is None:
        return

    record = {
        "source": source,
        "original": original,
        "raw_letters": raw_letters,
        "reason": decision.reason,
        "join_score": round(decision.join_score, 3),
        "segmented_score": round(decision.segmented_score, 3),
        "segmented_tokens": decision.segmented_tokens,
    }
    if context:
        record.update(context)
    reviews.append(record)


def collect_glyph_review(
    reviews: list[dict[str, Any]] | None,
    context: dict[str, Any] | None,
    token: str,
    original: str,
    reason: str,
) -> None:
    if reviews is None:
        return

    record = {
        "source": "glyph",
        "token": token,
        "original": original,
        "reason": reason,
    }
    if context:
        record.update(context)
    reviews.append(record)


def repair_uni_artifacts(
    text: str,
    scorer: LexicalScorer,
    reviews: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    if "/uni" not in text:
        return text

    parts: list[str] = []
    cursor = 0

    for match in UNI_TOKEN_RE.finditer(text):
        start, end = match.span()
        code = match.group(1).upper()
        if start < cursor:
            continue

        if code not in LIGATURE_UNI_MAP:
            replacement = chr(int(code, 16))
            parts.append(text[cursor:start])
            parts.append(replacement)
            cursor = end
            continue

        span = build_ligature_span(text, start, end, LIGATURE_UNI_MAP[code])
        if span.span_start < cursor:
            continue

        decision = scorer.decide_uni(span.raw_letters, span.original, span.token_context)
        parts.append(text[cursor:span.span_start])
        if decision.replacement is None:
            parts.append(span.original)
            collect_review(reviews, context, "uni", span.original, span.raw_letters, decision)
        else:
            parts.append(decision.replacement)
        cursor = span.span_end

    parts.append(text[cursor:])
    return "".join(parts)


def repair_glyph_artifacts(
    text: str,
    scorer: LexicalScorer,
    reviews: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    if "GLYPH<" not in text:
        return text

    parts: list[str] = []
    cursor = 0

    for match in GLYPH_TOKEN_RE.finditer(text):
        start, end = match.span()
        token = match.group(0)
        if start < cursor:
            continue

        direct = resolve_glyph_token(token)
        if direct is not None:
            parts.append(text[cursor:start])
            parts.append(direct)
            cursor = end
            continue

        if token == "GLYPH<14>" and is_degree_glyph_context(text, start, end):
            parts.append(text[cursor:start])
            parts.append("\u00b0")
            cursor = end
            continue

        ligature = GLYPH_LIGATURES.get(token)
        if ligature is None:
            parts.append(text[cursor:start])
            parts.append(token)
            collect_glyph_review(reviews, context, token, token, "unmapped_glyph")
            cursor = end
            continue

        span = build_ligature_span(text, start, end, ligature)
        if span.span_start < cursor:
            continue

        if token == "GLYPH<14>" and not (span.left or span.right):
            parts.append(text[cursor:start])
            parts.append(token)
            collect_glyph_review(reviews, context, token, token, "ambiguous_glyph14")
            cursor = end
            continue

        decision = scorer.decide_uni(span.raw_letters, span.original, span.token_context)
        parts.append(text[cursor:span.span_start])
        if decision.replacement is None:
            parts.append(span.original)
            collect_review(reviews, context, "glyph_ligature", span.original, span.raw_letters, decision)
        else:
            parts.append(decision.replacement)
        cursor = span.span_end

    parts.append(text[cursor:])
    return "".join(parts)


def repair_double_space_artifacts(
    text: str,
    scorer: LexicalScorer,
    reviews: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    if "  " not in text:
        return text

    parts: list[str] = []
    cursor = 0

    while True:
        match = DOUBLE_SPACE_RE.search(text, cursor)
        if match is None:
            parts.append(text[cursor:])
            break

        left, spaces, right = match.groups()
        window_start = max(0, match.start() - 40)
        window_end = min(len(text), match.end() + 40)
        window = text[window_start:window_end]
        decision = scorer.decide_double_space(left, right, window)

        parts.append(text[cursor:match.start()])
        if decision.replacement is None:
            if decision.status == "review":
                collect_review(reviews, context, "double_space", match.group(0), left + right, decision)
            parts.append(left)
            parts.append(spaces)
            cursor = match.start(3)
        else:
            parts.append(decision.replacement)
            cursor = match.end()

    return "".join(parts)


def fix_text(
    text: str,
    scorer: LexicalScorer | None = None,
    reviews: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    repair_double_spaces: bool = True,
) -> str:
    if not isinstance(text, str) or not text:
        return text

    scorer = scorer or LexicalScorer()
    repaired = text
    while True:
        updated = repair_uni_artifacts(repaired, scorer, reviews=reviews, context=context)
        if updated == repaired or not UNI_TOKEN_RE.search(updated):
            repaired = updated
            break
        repaired = updated
    while True:
        updated = repair_glyph_artifacts(repaired, scorer, reviews=reviews, context=context)
        if updated == repaired or not GLYPH_TOKEN_RE.search(updated):
            repaired = updated
            break
        repaired = updated
    if repair_double_spaces:
        repaired = repair_double_space_artifacts(repaired, scorer, reviews=reviews, context=context)
    return repaired


def count_ligature_tokens(value: str) -> int:
    if not isinstance(value, str):
        return 0
    total = len(UNI_LIGATURE_TOKEN_RE.findall(value))
    for token in GLYPH_LIGATURES:
        total += value.count(token)
    return total


def count_uni_tokens(value: str) -> int:
    return len(UNI_TOKEN_RE.findall(value)) if isinstance(value, str) else 0


def count_glyph_tokens(value: str) -> int:
    return len(GLYPH_TOKEN_RE.findall(value)) if isinstance(value, str) else 0


def repair_item_string(
    text: str,
    *,
    scorer: LexicalScorer,
    reviews: list[dict[str, Any]],
    context: dict[str, Any],
    allow_double_space_repair: bool,
) -> str:
    return fix_text(
        text,
        scorer=scorer,
        reviews=reviews,
        context=context,
        repair_double_spaces=allow_double_space_repair,
    )


def repair_table_text(
    table_rows: list[Any],
    *,
    scorer: LexicalScorer,
    reviews: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[Any], dict[str, int], list[dict[str, Any]]]:
    changed = False
    stats = {
        "table_cells_seen": 0,
        "table_cells_changed": 0,
        "items_with_ligature_tokens_before": 0,
        "items_with_ligature_tokens_after": 0,
        "items_with_uni_tokens_before": 0,
        "items_with_uni_tokens_after": 0,
        "items_with_glyph_tokens_before": 0,
        "items_with_glyph_tokens_after": 0,
        "double_space_changed": 0,
    }
    diagnostics: list[dict[str, Any]] = []
    repaired_rows: list[Any] = []

    for row_idx, row_dict in enumerate(table_rows):
        if not isinstance(row_dict, dict):
            repaired_rows.append(row_dict)
            continue

        repaired_row = dict(row_dict)
        for key, value in row_dict.items():
            if not isinstance(value, str):
                continue

            stats["table_cells_seen"] += 1
            stats["items_with_ligature_tokens_before"] += count_ligature_tokens(value)
            stats["items_with_uni_tokens_before"] += count_uni_tokens(value)
            stats["items_with_glyph_tokens_before"] += count_glyph_tokens(value)
            cell_context = {
                **context,
                "table_row_idx": row_idx,
                "table_key": key,
            }
            repaired_value = repair_item_string(
                value,
                scorer=scorer,
                reviews=reviews,
                context=cell_context,
                allow_double_space_repair=True,
            )
            stats["items_with_ligature_tokens_after"] += count_ligature_tokens(repaired_value)
            stats["items_with_uni_tokens_after"] += count_uni_tokens(repaired_value)
            stats["items_with_glyph_tokens_after"] += count_glyph_tokens(repaired_value)

            if repaired_value != value:
                if DOUBLE_SPACE_RE.search(value) and not DOUBLE_SPACE_RE.search(repaired_value):
                    stats["double_space_changed"] += 1
                repaired_row[key] = repaired_value
                changed = True
                stats["table_cells_changed"] += 1
                diagnostics.append(
                    {
                        "row_idx": context["row_idx"],
                        "page_no": context["page_no"],
                        "item_idx": context["item_idx"],
                        "label": context["label"],
                        "path": f"table[{row_idx!s}][{key}]",
                        "before": value,
                        "after": repaired_value,
                    }
                )

        repaired_rows.append(repaired_row)

    if not changed:
        return table_rows, stats, diagnostics
    return repaired_rows, stats, diagnostics


def process_pages_json(
    pages_json_str: str,
    row_idx: int,
    scorer: LexicalScorer,
    diagnostics: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    scope: str,
) -> tuple[str, dict[str, int]]:
    stats = {
        "items_seen": 0,
        "items_selected": 0,
        "items_changed": 0,
        "table_cells_seen": 0,
        "table_cells_changed": 0,
        "items_with_ligature_tokens_before": 0,
        "items_with_ligature_tokens_after": 0,
        "items_with_uni_tokens_before": 0,
        "items_with_uni_tokens_after": 0,
        "items_with_glyph_tokens_before": 0,
        "items_with_glyph_tokens_after": 0,
        "double_space_changed": 0,
    }

    if not isinstance(pages_json_str, str) or not pages_json_str or pages_json_str == "ANALYSIS_ERROR":
        return pages_json_str, stats

    try:
        pages = json.loads(pages_json_str)
    except (json.JSONDecodeError, TypeError):
        return pages_json_str, stats

    changed = False

    for page in pages:
        if not page:
            continue
        page_no = page.get("page_no", 0)
        items = page.get("items", [])
        for item_idx, item in enumerate(items):
            stats["items_seen"] += 1
            keep, reason = select_item_for_scope(item, page_no, scope)
            label = item.get("label", "")
            text = item.get("text", "")
            if not keep:
                continue

            stats["items_selected"] += 1
            context = {
                "row_idx": row_idx,
                "page_no": page_no,
                "item_idx": item_idx,
                "label": label,
                "selection_reason": reason,
                "scope": scope,
            }

            if isinstance(text, str):
                allow_double_space_repair = is_prose_like_label(label)
                stats["items_with_ligature_tokens_before"] += count_ligature_tokens(text)
                stats["items_with_uni_tokens_before"] += count_uni_tokens(text)
                stats["items_with_glyph_tokens_before"] += count_glyph_tokens(text)
                repaired = repair_item_string(
                    text,
                    scorer=scorer,
                    reviews=reviews,
                    context=context,
                    allow_double_space_repair=allow_double_space_repair,
                )
                stats["items_with_ligature_tokens_after"] += count_ligature_tokens(repaired)
                stats["items_with_uni_tokens_after"] += count_uni_tokens(repaired)
                stats["items_with_glyph_tokens_after"] += count_glyph_tokens(repaired)

                if repaired != text:
                    if DOUBLE_SPACE_RE.search(text) and not DOUBLE_SPACE_RE.search(repaired):
                        stats["double_space_changed"] += 1
                    item["text"] = repaired
                    changed = True
                    stats["items_changed"] += 1
                    diagnostics.append(
                        {
                            "row_idx": row_idx,
                            "page_no": page_no,
                            "item_idx": item_idx,
                            "label": label,
                            "path": "text",
                            "before": text,
                            "after": repaired,
                        }
                    )
            elif label == "table" and isinstance(text, list):
                repaired_table, table_stats, table_diagnostics = repair_table_text(
                    text,
                    scorer=scorer,
                    reviews=reviews,
                    context=context,
                )
                for key, value in table_stats.items():
                    stats[key] += value
                if repaired_table != text:
                    item["text"] = repaired_table
                    changed = True
                    stats["items_changed"] += 1
                    diagnostics.extend(table_diagnostics)

    if not changed:
        return pages_json_str, stats
    return json.dumps(pages, ensure_ascii=False), stats


def make_default_output_path(input_path: Path, suffix: str) -> Path:
    candidate = input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = input_path.with_name(f"{input_path.stem}{suffix}_{index}{input_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def resolve_output_path(input_path: Path, output_path: str | None, suffix: str) -> Path:
    if output_path is None:
        return make_default_output_path(input_path, suffix)

    path = Path(output_path)
    if path.resolve() == input_path.resolve():
        raise ValueError(f"Refusing to overwrite input file: {path}")
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing file: {path}")
    return path


def write_diagnostics(path: Path, diagnostics: list[dict[str, Any]], stats: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("fix_uni_tokens diagnostics\n")
        handle.write("=" * 80 + "\n")
        for key in sorted(stats):
            handle.write(f"{key}: {stats[key]}\n")
        handle.write("\n")

        for entry in diagnostics:
            handle.write(
                f"row={entry['row_idx']} page={entry['page_no']} item={entry['item_idx']} "
                f"label={entry['label']} path={entry['path']}\n"
            )
            handle.write(f"BEFORE: {entry['before']}\n")
            handle.write(f"AFTER:  {entry['after']}\n")
            handle.write("-" * 80 + "\n")


def write_reviews(path: Path, reviews: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(reviews, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair ligature artifacts in selected PagesJson item text.")
    parser.add_argument("input", help="Path to input feather file")
    parser.add_argument("--output", "-o", default=None, help="Path to output feather file")
    parser.add_argument("--diagnostics", "-d", default=None, help="Optional diagnostics text output")
    parser.add_argument("--review-output", default=None, help="Optional JSON review queue output")
    parser.add_argument(
        "--scope",
        choices=LIGATURE_SCOPE_CHOICES,
        default="non_reference",
        help="Which PagesJson items to clean (default: non_reference)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(input_path, args.output, "_unifixed")
    diagnostics_path = Path(args.diagnostics) if args.diagnostics else None
    review_path = Path(args.review_output) if args.review_output else None

    if diagnostics_path and diagnostics_path.exists():
        raise ValueError(f"Refusing to overwrite existing file: {diagnostics_path}")
    if review_path and review_path.exists():
        raise ValueError(f"Refusing to overwrite existing file: {review_path}")

    df = pd.read_feather(input_path)
    scorer = LexicalScorer(build_corpus_counts(df, args.scope))
    diagnostics: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    totals = Counter()

    repaired_pages: list[str] = []
    for row_idx, pages_json in enumerate(df["PagesJson"]):
        repaired, row_stats = process_pages_json(
            pages_json,
            row_idx,
            scorer,
            diagnostics,
            reviews,
            args.scope,
        )
        repaired_pages.append(repaired)
        totals.update(row_stats)

    output_df = df.copy()
    output_df["PagesJson"] = repaired_pages
    output_df.to_feather(output_path)

    if diagnostics_path:
        write_diagnostics(diagnostics_path, diagnostics, dict(totals))
    if review_path:
        write_reviews(review_path, reviews)

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Wrote repaired feather to {output_path}")
    print(f"scope={args.scope}")
    print(f"items_seen={totals['items_seen']}")
    print(f"items_selected={totals['items_selected']}")
    print(f"items_changed={totals['items_changed']}")
    print(f"table_cells_seen={totals['table_cells_seen']}")
    print(f"table_cells_changed={totals['table_cells_changed']}")
    print(f"items_with_ligature_tokens_before={totals['items_with_ligature_tokens_before']}")
    print(f"items_with_ligature_tokens_after={totals['items_with_ligature_tokens_after']}")
    print(f"items_with_uni_tokens_before={totals['items_with_uni_tokens_before']}")
    print(f"items_with_uni_tokens_after={totals['items_with_uni_tokens_after']}")
    print(f"items_with_glyph_tokens_before={totals['items_with_glyph_tokens_before']}")
    print(f"items_with_glyph_tokens_after={totals['items_with_glyph_tokens_after']}")
    print(f"double_space_changed={totals['double_space_changed']}")
    print(f"review_candidates={len(reviews)}")


if __name__ == "__main__":
    main()
