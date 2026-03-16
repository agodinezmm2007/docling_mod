# Topic Text Reconstruction

Post-processing script that reconstructs clean body text from `PagesJson` for topic modeling (e.g., LDA). Strips authors, affiliations, references, boilerplate, figures, tables, formulas, and other non-body content. Requires a feather file produced by one of the provenance extraction scripts (which populate the `PagesJson` column).

File: `scripts/topic_text_reconstruction.py`

## Input / Output

Input: a `.feather` file with a `PagesJson` column (from provenance extraction scripts).

```bash
python topic_text_reconstruction.py <input_feather> [--output <output_feather>] [--diagnostics <diagnostics_txt>]
```

Output:
- A `.feather` file with an added `TopicText` column containing the reconstructed body text. Default path: `<input_dir>/topic_text_output.feather`.
- Optionally, a diagnostics `.txt` file showing KEPT/DROP decisions for every item on pages 1 and 2 of each document, plus a preview of the first 1500 characters of `TopicText`.

Examples:

```bash
# basic usage, output to same directory as input
python topic_text_reconstruction.py output/results.feather

# specify output path
python topic_text_reconstruction.py output/results.feather --output output/with_topic_text.feather

# with diagnostics
python topic_text_reconstruction.py output/results.feather -o output/with_topic_text.feather -d output/diagnostics.txt
```

## Configuration Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `SKIP_LABELS` | `{"picture", "table", "caption", "formula", "page_footer", "footnote"}` | Docling item labels to always drop |
| `DROP_SECTION_HEADERS` | `True` | Whether to drop `section_header` items |
| `PAGE_FRONT_MATTER_MAX` | `2` | Apply front-matter filters (author, affiliation, charspan) on pages 1 through this value |
| `PAGE1_CHARSPAN_MIN` | `150` | On front-matter pages, drop items with charspan shorter than this |
| `MIN_ITEM_WORDS` | `3` | Drop items with fewer than this many words (2+ letter sequences) |

## Filter Pipeline

`extract_topic_text(pages_json_str)` iterates every item on every page and applies filters in this order. An item is dropped at the first matching filter.

```
item
  ├── is_reference? → drop
  ├── label in SKIP_LABELS? → drop
  ├── section_header and DROP_SECTION_HEADERS? → drop
  ├── non-string text? → drop
  ├── is_junk_text()? → drop
  ├── page_no <= PAGE_FRONT_MATTER_MAX?
  │     ├── is_author_line()? → drop
  │     ├── is_affiliation_line()? → drop
  │     └── charspan < PAGE1_CHARSPAN_MIN? → drop
  ├── is_boilerplate()? → drop
  ├── word_count < MIN_ITEM_WORDS? → drop
  └── KEPT → append to output
```

Items that pass all filters have their text stripped and joined with double newlines.

## Filter Functions

### `is_author_line(text)`

Detects author listing lines by separator patterns. Returns `True` if any of:

- Superscript digit patterns with 2+ middots or ampersands (`1 · Name · Name`)
- 2+ middot/bullet separators (`Name · Name · Name`)
- 2+ `/C1` separators (Docling encoding artifact for middot), unless line starts with "keywords"
- 2+ ampersand separators with no sentence verbs
- 2+ period separators (`Name . Name . Name`) with no sentence verbs
- Any `Æ` character (older encoding artifact)

Sentence verb check: looks for common verbs (`is`, `are`, `was`, `were`, `has`, `have`, `the`, `this`, `that`, `from`, `with`) to distinguish author lists from body text that happens to contain separators.

### `is_affiliation_line(text)`

Requires the line to start with an institutional keyword (`university`, `department`, `hospital`, `institute`, `school of`, `center for`, `college of`, `laboratory`, `faculty of`, `division of`, `observatory`, `research center`, `medical center`, `polytechnic`, `academy of`, or initials pattern). If so, confirms by:

- Postal code (4-6 digit number)
- Short text (<200 chars) with 2+ commas

### `is_junk_text(text)`

Drops layout artifacts:

- Empty or 3 characters or fewer
- No alphabetic characters
- Email addresses
- URLs or DOIs

### `is_boilerplate(text)`

Matches common boilerplate by line-start patterns:

- `Received:`, `Accepted:`, `Published online`, `Available online`
- Copyright lines (`(c) 2024`, `©`)
- `The author(s)`, `This article is published/distributed/licensed`
- `Open access`, `Electronic supplementary material`
- `Keywords`, `Abbreviations`, `E-mail:`, `Corresponding author`
- `Conflict of interest`, `Data availability`, `Funding`
- `Acknowledgment`, `Author contribution`
- `Supplementary data/material/information`

## Diagnostics

### `classify_item(item, page_no)`

Mirrors the filter logic in `extract_topic_text` but returns `(kept: bool, reason: str)` instead of silently skipping. Used by the diagnostics writer.

### `write_diagnostics(df, output_path)`

For each row, iterates pages 1 through `PAGE_FRONT_MATTER_MAX` and writes one line per item showing:

```
  [DROP (reason)                 ] cs= 260  text                 'Marie-Eve Héroux . H. Ross Anderson ...'
  [KEPT                          ] cs= 403  text                 'Objective Quantitative estimates ...'
```

Fields: keep/drop status with reason, charspan length, Docling label, first 500 characters of text. Followed by the first 1500 characters of `TopicText` for the row.

## Dependencies

- `pandas` for feather I/O
- `json` for parsing `PagesJson`
- `re` for all text pattern matching
- `argparse` for CLI
- Requires a feather file with `PagesJson` column from provenance extraction
