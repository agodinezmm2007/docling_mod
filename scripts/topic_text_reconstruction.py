# /mnt/c/Users/WSTATION/Desktop/docling_mods/scripts/topic_text_reconstruction.py
"""
topic_text_reconstruction.py

Reconstructs clean body text from PagesJson for topic modeling.
Strips authors, affiliations, references, boilerplate, figures, tables, etc.

Usage:
    python topic_text_reconstruction.py <input_feather> [--output <output_feather>] [--diagnostics <diagnostics_txt>]

Examples:
    python topic_text_reconstruction.py /path/to/pdf_df_output.feather
    python topic_text_reconstruction.py /path/to/pdf_df_output.feather --output /path/to/with_topic_text.feather
    python topic_text_reconstruction.py /path/to/pdf_df_output.feather --diagnostics /path/to/diagnostics.txt
"""

import argparse
import json
import re
import sys
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKIP_LABELS = {"picture", "table", "caption", "formula", "page_footer", "footnote"}
DROP_SECTION_HEADERS = True
PAGE_FRONT_MATTER_MAX = 2  # filter front-matter on pages 1 and 2
PAGE1_CHARSPAN_MIN = 150
MIN_ITEM_WORDS = 3


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def is_author_line(text):
    """Detect author listing lines by separator patterns."""
    # superscript numbers with middot/ampersand
    if re.search(r'(\d,\d|\d\s*·|\d\s*,\s*\d)', text):
        if len(re.findall(r'[·&]', text)) >= 2:
            return True
    # middot-separated
    if len(re.findall(r'\s[·•]\s', text)) >= 2:
        return True
    # /C1 separated (docling encoding artifact)
    if len(re.findall(r'/C1', text)) >= 2:
        if not re.match(r'(?i)^\s*keywords?\s', text):
            return True
    # ampersand-separated, no sentence verbs
    if len(re.findall(r'\s&\s', text)) >= 2:
        if not re.search(r'\b(is|are|was|were|has|have|the|this|that|from|with)\b', text.lower()):
            return True
    # period-separated: "Name . Name . Name"
    if len(re.findall(r'\s\.\s', text)) >= 2:
        if not re.search(r'\b(is|are|was|were|has|have|the|this|that|from|with)\b', text.lower()):
            return True
    # Æ separator (older encoding)
    if len(re.findall(r'Æ', text)) >= 1:
        return True
    return False


def is_affiliation_line(text):
    """Detect institutional affiliation lines by requiring affiliation-like start."""
    lower = text.lower().strip()

    starts_like_affiliation = bool(re.match(
        r'^(\d+\s+)?'
        r'('
        r'university|universit|institut|department|school of|'
        r'hospital|center for|centre for|college of|'
        r'laboratory|faculty of|division of|observatory|'
        r'research center|research centre|medical center|'
        r'medical centre|polytechnic|academy of|'
        r'[a-z]\.\s*(l\.\s*)?[a-z]'
        r')',
        lower
    ))
    if not starts_like_affiliation:
        return False

    # postal code confirms address
    if re.search(r'\b\d{4,6}\b', text):
        return True

    # short item with commas = likely affiliation without postal code
    if len(text) < 200 and text.count(',') >= 2:
        return True

    return False


def is_junk_text(text):
    """Filter out layout artifacts: lone punctuation, single chars, etc."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) <= 3:
        return True
    if not re.search(r'[a-zA-Z]', stripped):
        return True
    if re.match(r'^\s*\S+@\S+\s*$', stripped):
        return True
    if re.match(r'^\s*(https?://|doi:|www\.)\S+\s*$', stripped, re.I):
        return True
    return False


def is_boilerplate(text):
    """Catch common boilerplate that survives label filtering."""
    lower = text.lower().strip()
    patterns = [
        r'^received:?\s+\d',
        r'^accepted:?\s+\d',
        r'^published\s+online',
        r'^available\s+online',
        r'^\(?c\)?\s*\d{4}',
        r'^©',
        r'^the\s+author\(?s?\)?',
        r'^this\s+(article|work)\s+is\s+(published|distributed|licensed)',
        r'^open\s+access',
        r'^electronic\s+supplementary\s+material',
        r'^keywords?\s',
        r'^abbreviations?\s',
        r'^e-?mail:',
        r'^corresponding\s+author',
        r'^conflict\s+of\s+interest',
        r'^data\s+availability',
        r'^funding',
        r'^acknowledgment',
        r'^author\s+contribution',
        r'^supplementary\s+(data|material|information)',
    ]
    return any(re.match(p, lower) for p in patterns)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_topic_text(pages_json_str):
    """
    Reconstruct clean body text from PagesJson for topic modeling.
    No markdown, no tables, no figures, no references, no front-matter.
    """
    if not pages_json_str or pages_json_str == "ANALYSIS_ERROR":
        return ""

    try:
        pages = json.loads(pages_json_str)
    except (json.JSONDecodeError, TypeError):
        return ""

    body_parts = []

    for page in pages:
        if page is None:
            continue

        items = page.get("items", [])
        page_no = page.get("page_no", 0)

        for item in items:
            if item.get("is_reference", False):
                continue

            label = item.get("label", "")

            if label in SKIP_LABELS:
                continue

            if DROP_SECTION_HEADERS and label == "section_header":
                continue

            text = item.get("text", "")
            if not isinstance(text, str):
                continue

            if is_junk_text(text):
                continue

            # front-matter pages: author/affiliation patterns, charspan fallback
            if page_no <= PAGE_FRONT_MATTER_MAX:
                if is_author_line(text):
                    continue
                if is_affiliation_line(text):
                    continue
                charspan = item.get("charspan", [0, 0])
                span_len = charspan[1] - charspan[0] if len(charspan) == 2 else len(text)
                if span_len < PAGE1_CHARSPAN_MIN:
                    continue

            if is_boilerplate(text):
                continue

            word_count = len(re.findall(r'[a-zA-Z]{2,}', text))
            if word_count < MIN_ITEM_WORDS:
                continue

            body_parts.append(text.strip())

    return "\n\n".join(body_parts)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def classify_item(item, page_no):
    """Return (kept: bool, reason: str) for a single item."""
    label = item.get("label", "")
    text = item.get("text", "")
    if not isinstance(text, str):
        text = ""

    if item.get("is_reference", False):
        return False, "is_reference"
    if label in SKIP_LABELS:
        return False, f"label={label}"
    if DROP_SECTION_HEADERS and label == "section_header":
        return False, "section_header"
    if is_junk_text(text):
        return False, "junk"

    if page_no <= PAGE_FRONT_MATTER_MAX:
        if is_author_line(text):
            return False, "author_line"
        if is_affiliation_line(text):
            return False, "affiliation"
        charspan = item.get("charspan", [0, 0])
        span_len = charspan[1] - charspan[0] if len(charspan) == 2 else len(text)
        if span_len < PAGE1_CHARSPAN_MIN:
            return False, f"charspan={span_len}<{PAGE1_CHARSPAN_MIN}"

    if is_boilerplate(text):
        return False, "boilerplate"

    word_count = len(re.findall(r'[a-zA-Z]{2,}', text))
    if word_count < MIN_ITEM_WORDS:
        return False, f"words={word_count}<{MIN_ITEM_WORDS}"

    return True, ""


def write_diagnostics(df, output_path):
    """Write page 1+2 diagnostics + TopicText preview for every row."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            pages_str = row.get("PagesJson", "")
            if not pages_str or pages_str == "ANALYSIS_ERROR":
                f.write(f"row {row_idx}\n\nSKIPPED (no PagesJson)\n\n{'='*60}\n\n")
                continue

            pages = json.loads(pages_str)
            f.write(f"row {row_idx}\n\n")

            for pg in pages:
                if pg is None or not pg.get("items"):
                    continue
                page_no = pg.get("page_no", 0)
                if page_no < 1 or page_no > PAGE_FRONT_MATTER_MAX:
                    continue

                f.write(f"Page {page_no} items - KEPT vs DROPPED:\n")
                f.write("-" * 60 + "\n")

                for item in pg.get("items", []):
                    label = item["label"]
                    text = item.get("text", "")
                    if not isinstance(text, str):
                        text = f"[{type(text).__name__}]"
                    text_preview = text[:500]

                    kept, reason = classify_item(item, page_no)

                    charspan = item.get("charspan", [0, 0])
                    span_len = charspan[1] - charspan[0] if len(charspan) == 2 else len(text)
                    status = "KEPT" if kept else f"DROP ({reason})"
                    f.write(f"  [{status:30s}] cs={span_len:4d}  {label:20s} {text_preview!r}\n")

                f.write("\n")

            f.write(f"{'='*60}\n")
            f.write(f"ROW {row_idx}: First 1500 chars of TopicText\n")
            f.write(f"{'='*60}\n")
            f.write((row.get("TopicText", "") or "")[:1500])
            f.write(f"\n\n{'='*60}\n\n")

    print(f"Wrote diagnostics for {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reconstruct clean topic text from PagesJson.")
    parser.add_argument("input", help="Path to input feather file with PagesJson column")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output feather file (default: <input_dir>/topic_text_output.feather)")
    parser.add_argument("--diagnostics", "-d", default=None,
                        help="Path to diagnostics txt file (optional)")
    args = parser.parse_args()

    df = pd.read_feather(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    df["TopicText"] = df["PagesJson"].apply(extract_topic_text)

    for idx, row in df.iterrows():
        orig_words = len(row["FullText"].split()) if isinstance(row.get("FullText"), str) else 0
        topic_words = len(row["TopicText"].split()) if row["TopicText"] else 0
        orig_len = len(row["FullText"]) if isinstance(row.get("FullText"), str) else 0
        topic_len = len(row["TopicText"])
        reduction = (1 - topic_len / orig_len) * 100 if orig_len > 0 else 0
        print(f"Row {idx:2d}: FullText={orig_words:6d} words | TopicText={topic_words:6d} words | {reduction:.1f}% reduction")

    if args.output is None:
        from pathlib import Path
        args.output = str(Path(args.input).parent / "topic_text_output.feather")

    df.to_feather(args.output)
    print(f"Saved to {args.output}")

    if args.diagnostics:
        write_diagnostics(df, args.diagnostics)


if __name__ == "__main__":
    main()
