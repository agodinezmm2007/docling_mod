# Docling Extract Formulas Debug (with Provenance)

Three script variants that extract text, formulas, tables, and page-level provenance from academic PDFs using the Docling pipeline. The processing logic is identical across all three. They differ in GPU assignment strategy and whether debug layout images are generated.

| Script | GPU Mode | Layout Images | File |
|--------|----------|---------------|------|
| Multi-GPU | Round-robin across `GPU_IDS = [2, 4]` | No | `scripts/docling_extract_formulas_mp_multi_provenance.py` |
| Multi-GPU Debug | Round-robin across `GPU_IDS = [2, 4]` | Yes | `scripts/docling_extract_formulas_debug_mp_mult_provenance.py` |
| Single-GPU Debug | All workers share `GPU_ID = 4` | Yes | `scripts/docling_extract_formulas_debug_mp_singl_provenance.py` |

## Input / Output

Input: a folder of PDF files. The script scans for `*.pdf` and builds a DataFrame with `PDFPath` and `FileName` columns via `create_pdf_dataframe()`. Default folder: `scripts/sample_pdfs_cleaned/`.

Output: a timestamped `.feather` file in `scripts/output/` with columns:

| Column | Type | Content |
|--------|------|---------|
| `FullText` | `str` | Full markdown export of the document (`DoclingDocument.export_to_markdown()`) with `<!-- image -->` and `$$MALFORMED_FORMULA$$` artifacts stripped |
| `PagesJson` | `str` (JSON) | Per-page content, token counts, reference section detection, and item-level provenance. Table items contain rendered markdown from the table export. |
| `TablesJson` | `str` (JSON) | Per-table data as list-of-dicts with bounding box provenance |
| `EquationsJson` | `str` (JSON) | Per-formula LaTeX text with bounding box provenance |
| `TokenCount` | `int` | GPT-2 token count of `FullText` |
| `NumPages` | `int` | Number of pages in the document |
| `NumTables` | `int` | Number of tables detected |
| `NumPictures` | `int` | Number of pictures detected |
| `Error` | `str` or `None` | Error message if extraction failed, else `None` |

Each run also creates a timestamped log file in `scripts/logs/`. Debug variants write raw and postprocessed layout images to the debug output directory (defaults to `docling_debug/` at the repo root).

## Architecture

```
main process
  ├── scans PDF folder → creates DataFrame
  ├── ProcessPoolExecutor(max_workers=N, initializer=worker_initializer)
  │     ├── Worker 0 (GPU 2)
  │     │     ├── worker_initializer() → sets CUDA_VISIBLE_DEVICES, loads converter + tokenizer once
  │     │     ├── extract_pdf_with_docling(pdf_1) → converter.convert() → post-process → return dict
  │     │     ├── extract_pdf_with_docling(pdf_3) → reuses same converter
  │     │     └── ...
  │     ├── Worker 1 (GPU 4)
  │     │     └── ...
  │     └── ...
  └── collects results into DataFrame → saves timestamped .feather
```

Default `max_workers`: multi-GPU variants use `len(GPU_IDS) * 5` (10 for 2 GPUs). Single-GPU variant uses `5`.

## GPU Assignment

### Multi-GPU variants

```python
GPU_IDS = [2, 4]
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)
```

In `worker_initializer()`, each worker acquires `assign_lock`, reads `next_gpu.value % len(GPU_IDS)` to determine its GPU index, increments the counter, then sets `CUDA_VISIBLE_DEVICES` to `str(GPU_IDS[gpu_index])`. This produces round-robin assignment: worker 0 gets GPU 2, worker 1 gets GPU 4, worker 2 gets GPU 2, etc.

### Single-GPU variant

Skips the lock/counter mechanism and sets `CUDA_VISIBLE_DEVICES = str(GPU_ID)` for all workers.

## Worker Initialization

`worker_initializer()` runs once per worker process in all three variants. It:

1. Sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:8192"`
2. Configures logging to write to `scripts/logs/docling_testing.log`
3. Calls `init_pipeline()` (non-debug) or `init_debug_pipeline()` (debug variants) to create a `DocumentConverter` (stored in `_worker_converter`)
4. Calls `init_tokenizer()` to create a GPT-2 tokenizer via `tiktoken` (stored in `_worker_tokenizer`)

The converter and tokenizer are initialized once per worker and reused across all PDFs assigned to that worker.

## Pipeline Configuration

`init_debug_pipeline()` (or `init_pipeline()` for the non-debug variant) creates a `DocumentConverter` (see `documentation/docling_document_converter_reference.md` for the class API) with these settings:

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions, LayoutOptions,
    DOCLING_LAYOUT_HERON_101
)
from docling.datamodel.base_models import InputFormat
```

### AcceleratorOptions

See `documentation/docling_pipeline_options_reference.md`, section "AcceleratorOptions".

```python
AcceleratorOptions(num_threads=6, device=AcceleratorDevice.CUDA)
```

- `num_threads`: 6. CPU threads for model inference.
- `device`: `AcceleratorDevice.CUDA`. Forces GPU inference.

### PdfPipelineOptions

See `documentation/docling_pipeline_options_reference.md`, section "PdfPipelineOptions".

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = accelerator_options
pipeline_options.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_HERON_101)
pipeline_options.do_ocr = False
pipeline_options.do_formula_enrichment = True
pipeline_options.do_table_structure = True
pipeline_options.generate_page_images = True
pipeline_options.generate_parsed_pages = True
pipeline_options.images_scale = 2.0
```

- `do_ocr = False`: skips OCR. These are born-digital academic PDFs with embedded text.
- `do_formula_enrichment = True`: enables the SmolDocling-based formula/code extraction stage.
- `do_table_structure = True`: enables TableFormer table structure extraction.
- `generate_page_images = True`: generates PIL images of each page (needed by `code_formula_predictor.py` which calls `page.get_masked_image()`).
- `generate_parsed_pages = True`: keeps parsed cell data available for provenance extraction.
- `images_scale = 2.0`: renders pages at 2x resolution.
- `layout_options`: uses `DOCLING_LAYOUT_HERON_101`, a 76.7M parameter layout detection model.

### Debug Visualization (debug variants only)

```python
from docling.datamodel.settings import settings

debug_path = DEBUG_OUTPUT_PATH
if debug_path is None:
    debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docling_debug")
os.makedirs(debug_path, exist_ok=True)
settings.debug.visualize_raw_layout = True
settings.debug.visualize_layout = True
settings.debug.debug_output_path = debug_path
```

Saves layout bounding box visualizations to disk for inspection. The path is controlled by the module-level `DEBUG_OUTPUT_PATH` variable. When `None` (default), it resolves to `docling_debug/` at the repo root. Can be set three ways:

1. From the CLI (third positional argument).
2. From the notebook by setting `module.DEBUG_OUTPUT_PATH` before calling `do_docling_extraction()`.
3. Via the `output_dir` parameter of `do_docling_extraction()`, which sets `DEBUG_OUTPUT_PATH` internally before spawning workers.

The non-debug variant does not import `settings` or generate layout images.

### DocumentConverter

See `documentation/docling_document_converter_reference.md`, section "DocumentConverter".

```python
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Only PDF format is configured. The converter is created once per worker and reused for all PDFs assigned to that worker.

## Processing Pipeline per PDF

`extract_pdf_with_docling(pdf_path, idx, output_dir)` runs in a worker process. It uses the pre-initialized `_worker_converter` and `_worker_tokenizer`.

### Step 1: Document Conversion

```python
conv_res = converter.convert(pdf_path)
doc = conv_res.document  # DoclingDocument
```

`converter.convert()` returns a `ConversionResult` (see `documentation/docling_document_converter_reference.md`). The `document` attribute is a `DoclingDocument` (see `documentation/docling_document_reference.md`).

### Step 2: Markdown Export

```python
text_md = doc.export_to_markdown()
```

`DoclingDocument.export_to_markdown()` serializes the full document to markdown. See `documentation/docling_document_reference.md`, section "export_to_markdown" for all parameters. The script uses default parameters, then strips `<!-- image -->` and `$$MALFORMED_FORMULA$$` artifacts.

### Step 3: Table Export with Provenance (threaded)

Tables are exported before page metadata so their data can be injected into page content.

`_export_single_table(table, doc)` calls `table.export_to_dataframe(doc=doc)` (see `documentation/docling_document_reference.md`, section "TableItem.export_to_dataframe") and converts to `list[dict]` via `to_dict(orient="records")`. Adds provenance from `table.prov`.

Tables are exported in parallel using `ThreadPoolExecutor` with `min(POST_PROCESS_WORKERS, num_tables)` threads.

After export, a `tables_by_page` lookup dict is built mapping `page_no` to the list of table dicts for that page.

### Step 4a: Single-Pass Item Bucketing

`_build_all_page_items(doc)` iterates the document tree once using `doc.iterate_items(with_groups=False)` and buckets items by `page_no`.

`DoclingDocument.iterate_items()` yields `(NodeItem, int)` tuples (see `documentation/docling_document_reference.md`). Each item's `prov` attribute is a `list[ProvenanceItem]`, where each `ProvenanceItem` has `page_no`, `bbox` (`BoundingBox` with `l`, `t`, `r`, `b` coordinates), and `charspan`.

For each item, the function extracts:
- `label`: the `DocItemLabel` value (e.g., `"paragraph"`, `"formula"`, `"section_header"`)
- `text`: the item's text content
- `page_no`, `bbox`, `charspan`: from the first `ProvenanceItem`
- `is_reference`: initialized to `False`, set later during page metadata building

Only the first `ProvenanceItem` per item is used (the `break` after appending).

### Step 4b: Page Metadata Building (threaded)

`_build_page_metadata((page_no, page_items, count_tokens, page_tables))` runs in a `ThreadPoolExecutor` with `POST_PROCESS_WORKERS` threads (12). It operates on plain dicts, not `DoclingDocument` objects, so it is thread-safe.

For each page, it:
1. Injects table data into table items. `_match_table_to_item(item_bbox, page_tables)` finds the table whose provenance bbox center is closest to the item's bbox center. Matched table data (list of row dicts) replaces the item's `text` field.
2. Scans items for `section_header` labels containing "references", "bibliography", "works cited", or "literature cited". Once found, all subsequent items on that page are marked `is_reference = True`.
3. Builds page content by joining item texts. Table items (where `text` is a list of row dicts) are rendered as markdown tables via `_render_table_markdown()`. String items are included as-is. Strips artifacts.
4. Splits content into pre-reference and reference sections.
5. Computes token counts using the GPT-2 tokenizer for: full page, pre-reference content, reference content.

Returns a dict with keys: `page_no`, `content`, `content_before_references`, `reference_content`, `has_references`, `token_count`, `token_count_before_references`, `token_count_references`, `items`.

### Cross-Page Reference Tracking

After page metadata is built, a sequential pass across all pages propagates the `in_refs` flag. `_build_page_metadata` only detects references within a single page. This pass ensures that once a "References" section header is found on any page, all items on all subsequent pages are also marked as references. It recomputes `content_before_references`, `reference_content`, `has_references`, and associated token counts for affected pages.

### Step 5: Formula Extraction with Provenance

Iterates `doc.texts` (a `list[Union[TitleItem, SectionHeaderItem, ListItem, CodeItem, FormulaItem, TextItem]]`, see `documentation/docling_document_reference.md`). Filters for `TextItem` instances where `label == DocItemLabel.FORMULA` and `text != "$$MALFORMED_FORMULA$$"`.

For each formula, extracts:
- `latex`: the formula text
- `provenance`: list of `{page_no, bbox}` dicts from the item's `prov` list

### Result Assembly

The function returns a dict with all output columns. On exception, returns `"ANALYSIS_ERROR"` for text fields and `0` for counts.

## Helper Functions

### `_render_table_markdown(table_data)`

Takes a list of row dicts (from `table.export_to_dataframe().to_dict(orient="records")`) and renders a markdown table. Pipes and newlines in cell values are escaped.

### `_match_table_to_item(item_bbox, page_tables)`

Matches a document tree table item to its exported table data by finding the table whose provenance bbox center is closest (Euclidean distance) to the item's bbox center.

### `create_pdf_dataframe(pdf_folder)`

Scans `pdf_folder` for `*.pdf` files and returns a DataFrame with `PDFPath` and `FileName` columns.

## Reference Section Detection

`is_reference_section(text)` (standalone function, used for full-page detection) checks:
1. Header patterns: regex matches for "references", "bibliography", "works cited", "literature cited" as standalone lines.
2. Citation density: counts lines matching the pattern `^\s*\d+\.\s+[A-Z][^.]+\.\s+\(\d{4}\)` (numbered citations with year). If 3+ matches, classifies as references.

`_build_page_metadata` uses a simpler inline version: scans `section_header` items for reference-related keywords and flags all subsequent items on the same page as references. The cross-page reference tracking pass then propagates across page boundaries.

## Main Process Orchestration

`do_docling_extraction(df, max_workers, output_dir)` manages the `ProcessPoolExecutor`:

1. If `output_dir` is provided, sets the module-level `DEBUG_OUTPUT_PATH` (debug variants only).
2. Initializes output columns in the DataFrame.
3. Submits one `extract_pdf_with_docling` future per row. Skips rows with missing or nonexistent `PDFPath`.
4. Collects results via `as_completed()` and writes them back to the DataFrame.
5. Logs success/failure per row.

The `__main__` block:
1. Sets up timestamped logging (file in `scripts/logs/` + console output).
2. Scans a PDF folder (default: `scripts/sample_pdfs_cleaned/`, or first CLI argument).
3. Calls `do_docling_extraction()`.
4. Saves output `.feather` to `scripts/output/` (timestamped, or second CLI argument).
5. Debug variants accept a third CLI argument for the layout image output directory.

## Tuning Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `GPU_ID` / `GPU_IDS` | `4` / `[2, 4]` | GPU device IDs (line 17 in each script) |
| `num_threads` | `6` | CPU threads per worker for model inference |
| `POST_PROCESS_WORKERS` | `12` | Thread count for page metadata and table export |
| `max_workers` (multi) | `len(GPU_IDS) * 5` | Number of worker processes |
| `max_workers` (single) | `5` | Number of worker processes |
| `generate_page_images` | `True` | Renders page images for formula extraction |
| `DEBUG_OUTPUT_PATH` | `None` (resolves to `docling_debug/`) | Debug layout image output directory |

## Dependencies

- `docling.document_converter.DocumentConverter`, `PdfFormatOption` (see `documentation/docling_document_converter_reference.md`)
- `docling.datamodel.pipeline_options.PdfPipelineOptions`, `AcceleratorOptions`, `AcceleratorDevice`, `LayoutOptions` (see `documentation/docling_pipeline_options_reference.md`)
- `docling.datamodel.base_models.InputFormat` (see `documentation/docling_document_converter_reference.md`, section "InputFormat")
- `docling.datamodel.settings.settings` (debug variants only, for layout visualization)
- `docling_core.types.doc.TextItem`, `DocItemLabel` (see `documentation/docling_document_reference.md`)
- `tiktoken` for GPT-2 tokenization
- `pandas` for DataFrame I/O
- `multiprocessing`, `concurrent.futures` for parallelism
- `glob` for PDF folder scanning
