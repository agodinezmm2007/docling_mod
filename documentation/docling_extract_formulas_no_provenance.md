# Docling Extract Formulas (No Provenance)

Two script variants that extract text, formulas, and tables from academic PDFs using the Docling pipeline. These are simplified versions of the provenance scripts (see `documentation/docling_extract_formulas_debug_provenance.md`). They produce `FullText`, `TablesJson`, and `EquationsJson` but no page-level provenance (`PagesJson`), no bounding box data on formulas, no reference section detection, and no cross-page reference tracking.

Debug layout image generation is controlled by the `DEBUG_OUTPUT_PATH` variable rather than requiring separate debug scripts.

| Script | GPU Mode | File |
|--------|----------|------|
| Multi-GPU | Round-robin across `GPU_IDS = [2, 4]` | `scripts/docling_extract_formulas_mp_multi.py` |
| Single-GPU | All workers share `GPU_ID = 4` | `scripts/docling_extract_formulas_mp_singl.py` |

## Input / Output

Input: a folder of PDF files. The script scans for `*.pdf` and builds a DataFrame with `PDFPath` and `FileName` columns via `create_pdf_dataframe()`. Default folder: `scripts/sample_pdfs_cleaned/`.

Output: a timestamped `.feather` file in `scripts/output/` with columns:

| Column | Type | Content |
|--------|------|---------|
| `FullText` | `str` | Full markdown export of the document (`DoclingDocument.export_to_markdown()`) with `<!-- image -->` and `$$MALFORMED_FORMULA$$` artifacts stripped |
| `TablesJson` | `str` (JSON) | Per-table data as list-of-dicts (no bounding box provenance) |
| `EquationsJson` | `str` (JSON) | Per-formula LaTeX text (no bounding box provenance) |
| `TokenCount` | `int` | GPT-2 token count of `FullText` |
| `NumPages` | `int` | Number of pages in the document |
| `NumTables` | `int` | Number of tables detected |
| `NumPictures` | `int` | Number of pictures detected |
| `Error` | `str` or `None` | Error message if extraction failed, else `None` |

Each run also creates a timestamped log file in `scripts/logs/`.

## Differences from Provenance Scripts

| Feature | Provenance scripts | These scripts |
|---------|-------------------|---------------|
| `PagesJson` column | Yes (per-page items with bbox, charspan, reference flags, token counts) | No |
| `EquationsJson` bbox | Yes (`{latex, provenance: [{page_no, bbox}]}`) | No (`{latex}` only) |
| `TablesJson` bbox | Yes (`{data, provenance: [{page_no, bbox}]}`) | No (list-of-dicts only) |
| Table injection into page content | Yes (`_match_table_to_item`, `_render_table_markdown`) | No |
| Reference section detection | Yes (per-page + cross-page tracking) | No |
| `_build_all_page_items` | Yes | No |
| `_build_page_metadata` | Yes (threaded) | No |

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

Default `max_workers`: multi-GPU uses `len(GPU_IDS) * 5` (10 for 2 GPUs). Single-GPU uses `5`.

## GPU Assignment

### Multi-GPU variant

```python
GPU_IDS = [2, 4]
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)
```

Round-robin assignment in `worker_initializer()` via locked counter. Worker 0 gets GPU 2, worker 1 gets GPU 4, worker 2 gets GPU 2, etc.

### Single-GPU variant

`GPU_ID = 4`. All workers get `CUDA_VISIBLE_DEVICES = str(GPU_ID)`.

## Worker Initialization

`worker_initializer()` runs once per worker process. It:

1. Sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:8192"`
2. Configures logging to write to `scripts/logs/docling_testing.log`
3. Calls `init_pipeline()` to create a `DocumentConverter` (stored in `_worker_converter`)
4. Calls `init_tokenizer()` to create a GPT-2 tokenizer via `tiktoken` (stored in `_worker_tokenizer`)

## Pipeline Configuration

`init_pipeline()` creates a `DocumentConverter` with these settings:

```python
AcceleratorOptions(num_threads=6, device=AcceleratorDevice.CUDA)
LayoutOptions(model_spec=DOCLING_LAYOUT_HERON_101)

pipeline_options.do_ocr = False
pipeline_options.do_formula_enrichment = True
pipeline_options.do_table_structure = True
pipeline_options.generate_page_images = True
pipeline_options.generate_parsed_pages = True
pipeline_options.images_scale = 2.0
```

### Debug Visualization

Controlled by the module-level `DEBUG_OUTPUT_PATH` variable. When `None` (default), debug is disabled. When set to a path, the pipeline enables layout image generation:

```python
settings.debug.visualize_raw_layout = True
settings.debug.visualize_layout = True
settings.debug.debug_output_path = DEBUG_OUTPUT_PATH
```

Can be set three ways:

1. CLI third positional argument.
2. From the notebook: set `module.DEBUG_OUTPUT_PATH` before calling `do_docling_extraction()`.
3. Via the `output_dir` parameter of `do_docling_extraction()`.

## Processing Pipeline per PDF

`extract_pdf_with_docling(pdf_path, idx, output_dir)` runs in a worker process using the pre-initialized `_worker_converter` and `_worker_tokenizer`.

### Step 1: Document Conversion

```python
conv_res = converter.convert(pdf_path)
doc = conv_res.document
```

### Step 2: Markdown Export

```python
text_md = doc.export_to_markdown()
```

Strips `<!-- image -->` and `$$MALFORMED_FORMULA$$` artifacts.

### Step 3: Table Export (threaded)

`_export_single_table(table, doc)` calls `table.export_to_dataframe(doc=doc).to_dict(orient="records")`. Tables are exported in parallel using `ThreadPoolExecutor` with `min(POST_PROCESS_WORKERS, num_tables)` threads. No provenance/bbox is attached.

### Step 4: Formula Extraction

Iterates `doc.texts`, filters for `TextItem` with `label == DocItemLabel.FORMULA` and `text != "$$MALFORMED_FORMULA$$"`. Extracts `{"latex": el.text}` only (no bbox provenance).

### Result Assembly

Returns dict with `FullText`, `TablesJson`, `EquationsJson`, `TokenCount`, `NumPages`, `NumTables`, `NumPictures`, `Error`.

## Main Process Orchestration

`do_docling_extraction(df, max_workers, output_dir)`:

1. If `output_dir` is provided, sets `DEBUG_OUTPUT_PATH`.
2. Initializes output columns.
3. Submits one `extract_pdf_with_docling` future per row.
4. Collects results via `as_completed()`.

The `__main__` block:

```bash
# defaults: sample_pdfs_cleaned/, timestamped output
python docling_extract_formulas_mp_multi.py

# specify PDF folder
python docling_extract_formulas_mp_multi.py /path/to/pdfs

# specify PDF folder and output
python docling_extract_formulas_mp_multi.py /path/to/pdfs /path/to/output.feather

# enable debug layout images
python docling_extract_formulas_mp_multi.py /path/to/pdfs /path/to/output.feather /path/to/debug_images
```

## Tuning Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `GPU_ID` / `GPU_IDS` | `4` / `[2, 4]` | GPU device IDs (line 16 in each script) |
| `num_threads` | `6` | CPU threads per worker for model inference |
| `POST_PROCESS_WORKERS` | `12` | Thread count for table export |
| `max_workers` (multi) | `len(GPU_IDS) * 5` | Number of worker processes |
| `max_workers` (single) | `5` | Number of worker processes |
| `DEBUG_OUTPUT_PATH` | `None` | Set to a path to enable debug layout images |

## Dependencies

- `docling.document_converter.DocumentConverter`, `PdfFormatOption`
- `docling.datamodel.pipeline_options.PdfPipelineOptions`, `AcceleratorOptions`, `AcceleratorDevice`, `LayoutOptions`
- `docling.datamodel.base_models.InputFormat`
- `docling.datamodel.settings.settings` (only when `DEBUG_OUTPUT_PATH` is set)
- `docling_core.types.doc.TextItem`, `DocItemLabel`
- `tiktoken` for GPT-2 tokenization
- `pandas` for DataFrame I/O
- `multiprocessing`, `concurrent.futures` for parallelism
- `glob` for PDF folder scanning
