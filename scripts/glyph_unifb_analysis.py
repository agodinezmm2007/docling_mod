"""
Build a page/item-level inventory of raw glyph tokens from PagesJson.

The primary output is one row per token hit so the user can jump to the
corresponding PDF page and inspect the source text and bbox context.

Usage:
    python glyph_unifb_analysis.py <input_feather> [--output <hits_feather>]

Examples:
    python glyph_unifb_analysis.py output/results.feather
    python glyph_unifb_analysis.py output/results.feather --output output/results_glyph_hits.feather
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import requests


GLYPH_PATTERN = re.compile(r"GLYPH<[^>]+>")
UNI_PATTERN = re.compile(r"/uni[0-9A-Fa-f]{4}")
NONSPACE_PATTERN = re.compile(r"\S+")

TOKEN_STRIP_CHARS = ".,;:!?()[]{}<>\"'“”‘’`"


def classify_token(token: str) -> str:
    if token.startswith("GLYPH<"):
        inner = token[6:-1]
        if inner.isdigit():
            return "glyph_numeric"
        return "glyph_font_qualified"
    if token.startswith("/uniFB"):
        return "unifb"
    return "uni_other"


def _is_latin_extended_mojibake_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x0100 <= codepoint <= 0x024F
        or 0x1E00 <= codepoint <= 0x1EFF
    )


def _is_rare_marker_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x0370 <= codepoint <= 0x037F
        or 0x0208 <= codepoint <= 0x020F
    )


def classify_suspect_token(token: str) -> str | None:
    stripped = token.strip(TOKEN_STRIP_CHARS)
    if not stripped:
        return None

    if any(_is_rare_marker_char(char) for char in stripped):
        return "mojibake_rare_marker"

    latin_extended_count = sum(
        1 for char in stripped if _is_latin_extended_mojibake_char(char)
    )
    ascii_alpha_count = sum(
        1 for char in stripped if char.isascii() and char.isalpha()
    )
    alpha_total = latin_extended_count + ascii_alpha_count

    if (
        latin_extended_count >= 3
        and alpha_total > 0
        and (latin_extended_count / alpha_total) >= 0.3
    ):
        return "mojibake_latin_extended"

    if len(stripped) >= 4 and any(char in stripped for char in "\\$&"):
        ascii_graphic_count = sum(
            1 for char in stripped if char.isascii() and not char.isspace()
        )
        shifted_ascii_count = sum(
            1
            for char in stripped
            if char.isdigit() or ("A" <= char <= "Z") or char in "\\$&-"
        )
        if (
            ascii_graphic_count > 0
            and (shifted_ascii_count / ascii_graphic_count) >= 0.8
        ):
            return "mojibake_shifted_ascii"

    return None


def extract_doi(filename: str) -> str:
    name = filename.replace(".pdf", "")
    name = re.sub(r"_\d+$", "", name)
    name = name.replace("_", "/", 1)
    return name


def get_publisher(doi: str) -> str:
    logging.info("CrossRef lookup start doi=%s", doi)
    try:
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "docling_mods glyph_analysis (https://github.com/agodinezmm2007/docling_mod)"},
            timeout=15,
        )
        if response.status_code == 200:
            publisher = response.json()["message"].get("publisher", "Unknown")
            logging.info("CrossRef lookup success doi=%s publisher=%s", doi, publisher)
            return publisher
        logging.error("CrossRef lookup http_error doi=%s status=%s", doi, response.status_code)
        return "Unknown"
    except Exception as exc:
        logging.exception("CrossRef lookup exception doi=%s error=%s", doi, exc)
        return "Unknown"


def extract_context(text: str, start: int, end: int, radius: int = 40) -> tuple[str, str, str]:
    before = text[max(0, start - radius):start]
    after = text[end:min(len(text), end + radius)]
    snippet = text[max(0, start - radius):min(len(text), end + radius)]
    return before, after, snippet


def extract_token_hits_from_text(
    *,
    row_index,
    pdf_path: str,
    file_name: str,
    page_no,
    label: str,
    item_index: int,
    is_reference: bool,
    bbox,
    charspan,
    source_kind: str,
    table_column: str | None,
    text: str,
) -> list[dict]:
    hits = []
    seen_hits: set[tuple[int, int, str]] = set()

    def append_hit(start: int, end: int, token: str, token_type: str) -> None:
        key = (start, end, token_type)
        if key in seen_hits:
            return
        seen_hits.add(key)
        before, after, snippet = extract_context(text, start, end)
        hits.append(
            {
                "row_index": row_index,
                "PDFPath": pdf_path,
                "FileName": file_name,
                "page_no": page_no,
                "label": label,
                "item_index": item_index,
                "is_reference": bool(is_reference),
                "token": token,
                "token_type": token_type,
                "source_kind": source_kind,
                "table_column": table_column,
                "context_before": before,
                "context_after": after,
                "context_snippet": snippet,
                "bbox": json.dumps(bbox, ensure_ascii=False) if bbox is not None else None,
                "charspan": json.dumps(charspan, ensure_ascii=False) if charspan is not None else None,
            }
        )

    for pattern in (GLYPH_PATTERN, UNI_PATTERN):
        for match in pattern.finditer(text):
            token = match.group(0)
            append_hit(match.start(), match.end(), token, classify_token(token))

    for match in NONSPACE_PATTERN.finditer(text):
        token = match.group(0)
        token_type = classify_suspect_token(token)
        if token_type is not None:
            append_hit(match.start(), match.end(), token, token_type)

    return hits


def iter_hits_for_row(row_index, row: pd.Series) -> list[dict]:
    pdf_path = row.get("PDFPath", "")
    file_name = row.get("FileName", "")
    pages_json = row.get("PagesJson")
    if not isinstance(pages_json, str) or not pages_json:
        return []

    try:
        pages = json.loads(pages_json)
    except Exception:
        return []

    hits: list[dict] = []
    for page in pages if isinstance(pages, list) else []:
        page_no = page.get("page_no")
        items = page.get("items", []) or []
        for item_index, item in enumerate(items):
            label = item.get("label", "unknown")
            is_reference = item.get("is_reference", False)
            bbox = item.get("bbox")
            charspan = item.get("charspan")
            text = item.get("text")

            if isinstance(text, str):
                hits.extend(
                    extract_token_hits_from_text(
                        row_index=row_index,
                        pdf_path=pdf_path,
                        file_name=file_name,
                        page_no=page_no,
                        label=label,
                        item_index=item_index,
                        is_reference=is_reference,
                        bbox=bbox,
                        charspan=charspan,
                        source_kind="item_text",
                        table_column=None,
                        text=text,
                    )
                )
            elif isinstance(text, list):
                for row_dict in text:
                    if not isinstance(row_dict, dict):
                        continue
                    for column_name, value in row_dict.items():
                        if not isinstance(value, str):
                            continue
                        hits.extend(
                            extract_token_hits_from_text(
                                row_index=row_index,
                                pdf_path=pdf_path,
                                file_name=file_name,
                                page_no=page_no,
                                label=label,
                                item_index=item_index,
                                is_reference=is_reference,
                                bbox=bbox,
                                charspan=charspan,
                                source_kind="table_cell",
                                table_column=str(column_name),
                                text=value,
                            )
                        )
    return hits


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    hits: list[dict] = []
    for row_index, row in df.iterrows():
        hits.extend(iter_hits_for_row(row_index, row))

    columns = [
        "row_index",
        "PDFPath",
        "FileName",
        "page_no",
        "label",
        "item_index",
        "is_reference",
        "token",
        "token_type",
        "source_kind",
        "table_column",
        "context_before",
        "context_after",
        "context_snippet",
        "bbox",
        "charspan",
    ]
    result = pd.DataFrame(hits, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        by=["FileName", "page_no", "item_index", "token"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_hits(hits_df: pd.DataFrame) -> pd.DataFrame:
    if hits_df.empty:
        return pd.DataFrame(
            columns=["FileName", "page_no", "token", "token_type", "hit_count", "labels", "source_kinds"]
        )

    summary = (
        hits_df.groupby(["FileName", "page_no", "token", "token_type"], dropna=False)
        .agg(
            hit_count=("token", "size"),
            labels=("label", lambda values: ",".join(sorted(set(str(v) for v in values)))),
            source_kinds=("source_kind", lambda values: ",".join(sorted(set(str(v) for v in values)))),
        )
        .reset_index()
        .sort_values(by=["FileName", "page_no", "token"], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def build_summary_json(hits_df: pd.DataFrame) -> list[dict]:
    summary_df = summarize_hits(hits_df)
    if summary_df.empty:
        return []

    files: list[dict] = []
    for file_name, file_group in summary_df.groupby("FileName", sort=False, dropna=False):
        pages: list[dict] = []
        for page_no, page_group in file_group.groupby("page_no", sort=False, dropna=False):
            tokens = []
            for _, row in page_group.iterrows():
                tokens.append(
                    {
                        "token": row["token"],
                        "token_type": row["token_type"],
                        "hit_count": int(row["hit_count"]),
                        "labels": row["labels"],
                        "source_kinds": row["source_kinds"],
                    }
                )
            pages.append(
                {
                    "page_no": None if pd.isna(page_no) else int(page_no),
                    "token_count": len(tokens),
                    "tokens": tokens,
                }
            )

        files.append(
            {
                "FileName": file_name,
                "page_count": len(pages),
                "pages": pages,
            }
        )
    return files


def build_default_output_path(input_path: str) -> Path:
    path = Path(input_path)
    return path.parent / f"{path.stem}_glyph_hits.feather"


def build_default_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_pages.json")


def build_default_subset_path(input_path: str) -> Path:
    path = Path(input_path)
    return path.parent / f"{path.stem}_glyph_unifb_subset.feather"


def build_subset_dataframe(input_df: pd.DataFrame, hits_df: pd.DataFrame) -> pd.DataFrame:
    subset_columns = [
        "has_glyph",
        "has_unifb",
        "glyph_count",
        "unifb_count",
        "unique_glyph_tokens",
        "unique_uni_tokens",
        "has_mojibake",
        "mojibake_count",
        "unique_mojibake_tokens",
        "doi",
        "publisher",
        "total_glyph_unifb_count",
        "total_suspect_count",
    ]
    if hits_df.empty:
        result = input_df.iloc[0:0].copy()
        for col in subset_columns:
            result[col] = []
        return result

    subset = input_df.loc[hits_df["row_index"].drop_duplicates().tolist()].copy().reset_index(drop=True)
    grouped = hits_df.groupby("row_index", sort=False)
    records = []
    publisher_cache: dict[str, str] = {}
    for row_index, group in grouped:
        file_name = input_df.at[row_index, "FileName"]
        doi = extract_doi(file_name)
        if doi not in publisher_cache:
            publisher_cache[doi] = get_publisher(doi)
        publisher = publisher_cache[doi]
        glyph_tokens = sorted(set(group.loc[group["token_type"].str.startswith("glyph"), "token"].tolist()))
        uni_tokens = sorted(set(group.loc[group["token_type"].isin(["unifb", "uni_other"]), "token"].tolist()))
        mojibake_tokens = sorted(
            set(group.loc[group["token_type"].str.startswith("mojibake"), "token"].tolist())
        )
        records.append(
            {
                "row_index": row_index,
                "has_glyph": bool((group["token_type"].str.startswith("glyph")).any()),
                "has_unifb": bool((group["token_type"] == "unifb").any()),
                "glyph_count": int(group["token_type"].str.startswith("glyph").sum()),
                "unifb_count": int((group["token_type"] == "unifb").sum()),
                "unique_glyph_tokens": ",".join(glyph_tokens),
                "unique_uni_tokens": ",".join(uni_tokens),
                "has_mojibake": bool((group["token_type"].str.startswith("mojibake")).any()),
                "mojibake_count": int(group["token_type"].str.startswith("mojibake").sum()),
                "unique_mojibake_tokens": ",".join(mojibake_tokens),
                "doi": doi,
                "publisher": publisher,
                "total_glyph_unifb_count": int(len(group)),
                "total_suspect_count": int(len(group)),
            }
        )

    records_df = pd.DataFrame(records).sort_values(by="row_index", kind="stable").reset_index(drop=True)
    for col in subset_columns:
        subset[col] = records_df[col]
    return subset


def print_summary(hits_df: pd.DataFrame) -> None:
    if hits_df.empty:
        print("No GLYPH, /uni, or suspect mojibake tokens found in PagesJson.")
        return

    token_type_counts = hits_df["token_type"].value_counts().to_dict()
    rows_with_hits = hits_df["row_index"].nunique()
    pages_with_hits = hits_df[["FileName", "page_no"]].drop_duplicates().shape[0]
    files_with_hits = hits_df["FileName"].nunique()

    print("\nSummary:")
    print(f"  Files with hits: {files_with_hits}")
    print(f"  Rows with hits: {rows_with_hits}")
    print(f"  Pages with hits: {pages_with_hits}")
    print(f"  Total hit rows: {len(hits_df)}")
    print(f"  Unique tokens: {hits_df['token'].nunique()}")
    for token_type, count in sorted(token_type_counts.items()):
        print(f"  {token_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GLYPH, /uni, and suspect mojibake tokens in PagesJson and emit page/item-level hit rows.")
    parser.add_argument("input", help="Path to input feather file with PagesJson column")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output hit-level feather file (default: <input_stem>_glyph_hits.feather)",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON path for grouped file/page/token summary (default: <output_stem>_pages.json)",
    )
    parser.add_argument(
        "--subset-output",
        default=None,
        help="Optional feather path for article-level subset output (default: <input_stem>_glyph_unifb_subset.feather)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    df = pd.read_feather(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    hits_df = analyze_dataframe(df)
    output_path = Path(args.output) if args.output else build_default_output_path(args.input)
    summary_path = Path(args.summary_output) if args.summary_output else build_default_summary_path(output_path)
    subset_path = Path(args.subset_output) if args.subset_output else build_default_subset_path(args.input)

    if hits_df.empty:
        hits_df.to_feather(output_path)
        summary_path.write_text(json.dumps(build_summary_json(hits_df), indent=2), encoding="utf-8")
        build_subset_dataframe(df, hits_df).to_feather(subset_path)
        print(f"Saved empty hit inventory to {output_path}")
        print(f"Saved empty page summary to {summary_path}")
        print(f"Saved empty subset feather to {subset_path}")
        return

    hits_df.to_feather(output_path)
    summary_path.write_text(json.dumps(build_summary_json(hits_df), indent=2), encoding="utf-8")
    build_subset_dataframe(df, hits_df).to_feather(subset_path)
    print(f"Saved {len(hits_df)} hit rows to {output_path}")
    print(f"Saved page summary to {summary_path}")
    print(f"Saved subset feather to {subset_path}")
    print_summary(hits_df)


if __name__ == "__main__":
    main()
