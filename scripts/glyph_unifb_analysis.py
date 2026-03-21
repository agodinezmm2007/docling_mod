# glyph_unifb_analysis.py
"""
Scans extracted FullText for GLYPH<N> and /uniFBxx tokens, looks up publishers
via CrossRef, and outputs a subset feather with diagnostic columns.

Usage:
    python glyph_unifb_analysis.py <input_feather> [--output <output_feather>]

Examples:
    python glyph_unifb_analysis.py output/3_16_2026_198_recleaned.feather
    python glyph_unifb_analysis.py output/results.feather --output output/glyph_subset.feather
"""

import argparse
import re
import time
import pandas as pd
import requests


GLYPH_PATTERN = re.compile(r'GLYPH<[^>]+>')
UNIFB_PATTERN = re.compile(r'/uniFB\w+')


def extract_doi(filename):
    """Extract DOI from filename. e.g. '10.1016_j.chest.2021.07.2170_889.pdf' -> '10.1016/j.chest.2021.07.2170'"""
    name = filename.replace('.pdf', '')
    name = re.sub(r'_\d+$', '', name)
    name = name.replace('_', '/', 1)
    return name


def get_publisher(doi):
    """Look up publisher from CrossRef API."""
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "docling_mods glyph_analysis (https://github.com/agodinezmm2007/docling_mod)"},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()["message"].get("publisher", "Unknown")
        return "Unknown"
    except Exception:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Analyze GLYPH and /uniFB tokens in extracted text.")
    parser.add_argument("input", help="Path to input feather file with FullText column")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output feather file (default: <input_dir>/glyph_unifb_subset.feather)")
    args = parser.parse_args()

    df = pd.read_feather(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # scan for tokens
    indices = []
    records = []
    for idx, row in df.iterrows():
        text = row['FullText'] if isinstance(row['FullText'], str) else ''
        glyphs = GLYPH_PATTERN.findall(text)
        unifbs = UNIFB_PATTERN.findall(text)
        if not glyphs and not unifbs:
            continue
        indices.append(idx)
        doi = extract_doi(row['FileName'])
        records.append({
            'has_glyph': len(glyphs) > 0,
            'has_unifb': len(unifbs) > 0,
            'glyph_count': len(glyphs),
            'unifb_count': len(unifbs),
            'unique_glyph_unifb': ','.join(sorted(set(glyphs + unifbs))),
            'doi': doi,
            'total_glyph_unifb_count': len(glyphs) + len(unifbs),
        })

    print(f"Found {len(records)} rows with GLYPH or /uniFB tokens")

    if not records:
        print("No tokens found. Nothing to output.")
        return

    # look up publishers from CrossRef
    print(f"Looking up publishers via CrossRef for {len(records)} DOIs...")
    for i, r in enumerate(records):
        r['publisher'] = get_publisher(r['doi'])
        print(f"  [{i+1}/{len(records)}] {r['doi']} -> {r['publisher']}")
        time.sleep(0.1)  

    # build output dataframe
    result = df.iloc[indices].copy().reset_index(drop=True)
    for i, r in enumerate(records):
        for col, val in r.items():
            result.at[i, col] = val

    if args.output is None:
        from pathlib import Path
        args.output = str(Path(args.input).parent / "glyph_unifb_subset.feather")

    result.to_feather(args.output)
    print(f"\nSaved {len(result)} rows to {args.output}")

    # summary
    print(f"\nSummary:")
    print(f"  Rows with GLYPH tokens: {sum(r['has_glyph'] for r in records)}")
    print(f"  Rows with /uniFB tokens: {sum(r['has_unifb'] for r in records)}")
    print(f"  Total GLYPH occurrences: {sum(r['glyph_count'] for r in records)}")
    print(f"  Total /uniFB occurrences: {sum(r['unifb_count'] for r in records)}")
    print(f"\n  Publishers:")
    pub_counts = {}
    for r in records:
        pub_counts[r['publisher']] = pub_counts.get(r['publisher'], 0) + 1
    for pub, count in sorted(pub_counts.items(), key=lambda x: -x[1]):
        print(f"    {pub}: {count}")


if __name__ == "__main__":
    main()
