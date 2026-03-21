# docling_extract_formulas_mp_multi_provenance.py

import argparse
import os
import time
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from functools import partial
from pathlib import Path
import pandas as pd
import logging
import json
import tiktoken
import re
import glob
from typing import Any
from pydantic import BaseModel, ConfigDict, ValidationError
# import any other modules that are GPU related only after the workers are initialized

# list of GPU IDs you want to use:
GPU_IDS = [2, 4]

# global variables for assigning GPUs
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)

# per-worker globals (initialized once in worker_initializer, reused across PDFs)
_worker_converter = None
_worker_tokenizer = None

OUTPUT_COLUMNS = [
    "PDFPath",
    "FileName",
    "FullText",
    "PagesJson",
    "TablesJson",
    "EquationsJson",
    "TokenCount",
    "NumPages",
    "NumTables",
    "NumPictures",
    "Error",
]
RESULT_COLUMNS = OUTPUT_COLUMNS[2:]


class ExtractionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PDFPath: str
    FileName: str
    FullText: str
    PagesJson: str
    TablesJson: str
    EquationsJson: str
    TokenCount: int
    NumPages: int
    NumTables: int
    NumPictures: int
    Error: str | None

def worker_initializer():
    """
    This initializer is called once per worker process.
    It uses a global counter to assign one GPU from GPU_IDS to each worker,
    then loads the converter and tokenizer once for reuse.
    """
    global _worker_converter, _worker_tokenizer

    with assign_lock, next_gpu.get_lock():
        # determine which GPU to assign based on a round-robin strategy
        gpu_index = next_gpu.value % len(GPU_IDS)
        next_gpu.value += 1
    # IMPORTANT: Set CUDA_VISIBLE_DEVICES early!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_index])
    # set other CUDA-related env vars as needed
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    # configure logging for worker process
    import logging
    LOG_FILE = "/mnt/c/Users/WSTATION/Desktop/docling_mods/scripts/logs/docling_testing.log"

    # clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    # add file handler
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)

    logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")

    # load converter and tokenizer once per worker
    _worker_converter = init_pipeline()
    _worker_tokenizer = init_tokenizer()
    logging.info(f"Worker {multiprocessing.current_process().name} converter and tokenizer initialized")

def init_tokenizer():
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        return lambda text: len(tokenizer.encode(text)) if text else 0
    except ImportError:
        return lambda text: 0

def init_pipeline():
    """
    Called once in each worker process to set environment variables and
    create a DocumentConverter.
    Note: Since we already set CUDA_VISIBLE_DEVICES in the initializer,
    we do not need to reset it here.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        AcceleratorDevice,
        PdfPipelineOptions,
        LayoutOptions,
        DOCLING_LAYOUT_HERON_101  # 76.7M parameter model
    )
    from docling.datamodel.base_models import InputFormat

    # adjust additional environment settings if required
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA
    )

    # configure layout model
    layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON_101
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.layout_options = layout_options
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True # formula extraction pipeline needs generate_page_images = True because code_formula_predictor.py calls page.get_masked_image()
    pipeline_options.generate_parsed_pages = True # generate_parsed_pages = True keeps the parsed cell data available, which feeds into provenance extraction 
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    return converter

def is_reference_section(text):
    """
    Detect if page content is a references/bibliography section.
    """
    if not text or len(text.strip()) < 50:
        return False

    text_lower = text.lower()

    # check for reference section headers
    header_patterns = [
        r'^\s*references?\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*works\s+cited\s*$',
        r'^\s*literature\s+cited\s*$',
    ]

    for pattern in header_patterns:
        if re.search(pattern, text_lower, re.MULTILINE):
            return True

    # check for citation density (multiple numbered citations)
    citation_pattern = r'^\s*\d+\.\s+[A-Z][^.]+\.\s+\(\d{4}\)'
    citation_matches = re.findall(citation_pattern, text, re.MULTILINE)

    # if more than 3 citations on a page, likely references
    if len(citation_matches) >= 3:
        return True

    return False


def _build_all_page_items(doc):
    """
    Single-pass iteration over the entire document tree.
    Buckets items by page_no, avoiding N separate full-tree scans.
    Returns dict[int, list[dict]] mapping page_no -> list of item dicts.
    """
    page_items_map = {}

    for item, level in doc.iterate_items(with_groups=False):
        if not hasattr(item, 'prov') or not item.prov:
            continue
        for prov in item.prov:
            page_no = prov.page_no
            item_label = item.label.value if hasattr(item, 'label') else "unknown"
            item_text = item.text if hasattr(item, 'text') else ""

            item_data = {
                "label": item_label,
                "text": item_text,
                "page_no": prov.page_no,
                "bbox": {
                    "l": prov.bbox.l,
                    "t": prov.bbox.t,
                    "r": prov.bbox.r,
                    "b": prov.bbox.b
                } if prov.bbox else None,
                "charspan": list(prov.charspan) if prov.charspan else None,
                "is_reference": False  # set during per-page post-processing
            }

            if page_no not in page_items_map:
                page_items_map[page_no] = []
            page_items_map[page_no].append(item_data)
            break  # only take first prov per item

    return page_items_map


def _render_table_markdown(table_data):
    """Render a list of row dicts as a markdown table."""
    if not table_data:
        return ""
    cols = list(table_data[0].keys())
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in table_data:
        cells = [str(row.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _match_table_to_item(item_bbox, page_tables):
    """
    Find the table from page_tables whose provenance bbox best matches the item bbox.
    Match by overlap: the table whose bbox center is closest to the item bbox center.
    Returns the matched table dict or None.
    """
    if not page_tables or not item_bbox:
        return None

    item_cx = (item_bbox["l"] + item_bbox["r"]) / 2
    item_cy = (item_bbox["t"] + item_bbox["b"]) / 2

    best_table = None
    best_dist = float("inf")
    for t in page_tables:
        for prov in t.get("provenance", []):
            tb = prov.get("bbox")
            if not tb:
                continue
            t_cx = (tb["l"] + tb["r"]) / 2
            t_cy = (tb["t"] + tb["b"]) / 2
            dist = (item_cx - t_cx) ** 2 + (item_cy - t_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_table = t
    return best_table


def _build_page_metadata(args):
    """
    Build page metadata from pre-bucketed items. No doc access needed.
    Safe for ThreadPoolExecutor since it only operates on plain dicts/strings.
    page_tables is a list of table dicts from TablesJson for this page.
    """
    page_no, page_items, count_tokens, page_tables = args

    # populate table items with their extracted data from TablesJson
    used_tables = set()
    for item in page_items:
        if item["label"] == "table":
            matched = _match_table_to_item(item.get("bbox"), page_tables)
            if matched and id(matched) not in used_tables:
                used_tables.add(id(matched))
                item["text"] = matched.get("data", [])

    # detect references section
    in_references = False
    for item in page_items:
        if item["label"] == "section_header":
            text_lower = item["text"].lower().strip()
            if any(term in text_lower for term in
                   ['references', 'bibliography', 'works cited', 'literature cited']):
                in_references = True
        item["is_reference"] = in_references

    # build page content from item texts, rendering table data as markdown
    all_texts = []
    for item in page_items:
        if item["label"] == "table":
            logging.debug(f"Page {page_no}: table item text type={type(item['text']).__name__}, len={len(item['text']) if isinstance(item['text'], (list, str)) else 'N/A'}")
        if isinstance(item["text"], list):
            md = _render_table_markdown(item["text"])
            logging.debug(f"Page {page_no}: rendered table markdown, len={len(md)}")
            if md:
                all_texts.append(md)
        elif isinstance(item["text"], str) and item["text"]:
            all_texts.append(item["text"])
    page_md = "\n\n".join(all_texts)

    # clean artifacts
    for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
        page_md = page_md.replace(artifact, "")

    # separate content before and after references
    content_before_refs = []
    reference_content = []
    for item in page_items:
        if isinstance(item["text"], list):
            text = _render_table_markdown(item["text"])
        elif isinstance(item["text"], str):
            text = item["text"]
        else:
            text = ""
        if not text:
            continue
        if item["is_reference"]:
            reference_content.append(text)
        else:
            content_before_refs.append(text)

    content_before_refs_text = "\n\n".join(content_before_refs)
    reference_content_text = "\n\n".join(reference_content)
    has_references = len(reference_content) > 0

    # token counts
    token_count = count_tokens(page_md)
    token_count_before_refs = count_tokens(content_before_refs_text) if content_before_refs_text else 0
    token_count_refs = count_tokens(reference_content_text) if reference_content_text else 0

    return {
        "page_no": page_no,
        "content": page_md,
        "content_before_references": content_before_refs_text,
        "reference_content": reference_content_text,
        "has_references": has_references,
        "token_count": token_count,
        "token_count_before_references": token_count_before_refs,
        "token_count_references": token_count_refs,
        "items": page_items
    }


def _export_single_table(table, doc):
    """Worker function for threaded table export."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrame columns are not unique")
        table_data = {
            "data": table.export_to_dataframe(doc=doc).to_dict(orient="records")
        }
        if table.prov:
            table_data["provenance"] = [
                {
                    "page_no": prov.page_no,
                    "bbox": {
                        "l": prov.bbox.l,
                        "t": prov.bbox.t,
                        "r": prov.bbox.r,
                        "b": prov.bbox.b
                    } if prov.bbox else None
                }
                for prov in table.prov
            ]
        return table_data


# number of threads for post-processing (page metadata, tables, etc.)
POST_PROCESS_WORKERS = 12


def extract_pdf_with_docling(pdf_path: str, idx: int, output_dir=None) -> dict:
    """
    In the child process, convert the PDF and return results with page-level provenance.
    Post-processing uses ThreadPoolExecutor for parallelism within a single PDF.
    """
    global _worker_converter, _worker_tokenizer
    t_total_start = time.time()
    logging.info(f"[Row {idx}] Starting extraction for {pdf_path}")

    converter = _worker_converter
    count_tokens = _worker_tokenizer

    try:
        # step 3: convert document (GPU-accelerated)
        t0 = time.time()
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)
        logging.info(f"[Row {idx}] Step 1/6: Docling conversion done in {time.time()-t0:.2f}s — {num_pages} pages, {num_tables} tables, {num_pictures} pictures")

        # step 4: full markdown export
        t0 = time.time()
        text_md = doc.export_to_markdown()
        artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
        for artifact in artifacts_to_remove:
            text_md = text_md.replace(artifact, "")
        logging.info(f"[Row {idx}] Step 2/6: Full markdown export done in {time.time()-t0:.2f}s ({len(text_md)} chars)")

        # step 3: export tables with provenance (threaded, must happen before page metadata)
        t0 = time.time()
        all_tables_json = []
        if num_tables > 0:
            table_export_fn = partial(_export_single_table, doc=doc)
            with ThreadPoolExecutor(max_workers=min(POST_PROCESS_WORKERS, num_tables)) as executor:
                futures = {
                    executor.submit(table_export_fn, table): t_idx
                    for t_idx, table in enumerate(doc.tables)
                }
                table_results = [None] * num_tables
                for future in as_completed(futures):
                    t_idx = futures[future]
                    try:
                        table_results[t_idx] = future.result()
                    except Exception as e:
                        logging.error(f"[Row {idx}] Table {t_idx} export failed: {e}", exc_info=True)
                        table_results[t_idx] = {"data": [], "error": str(e)}
                all_tables_json = table_results
        logging.info(f"[Row {idx}] Step 3/6: Table export done in {time.time()-t0:.2f}s ({num_tables} tables)")

        # build table lookup by page_no for injection into page metadata
        tables_by_page = {}
        for t in all_tables_json:
            if not t:
                continue
            for prov in t.get("provenance", []):
                pg = prov.get("page_no")
                if pg is not None:
                    if pg not in tables_by_page:
                        tables_by_page[pg] = []
                    tables_by_page[pg].append(t)

        # step 4a: single-pass iterate_items -> bucket by page (one tree traversal)
        t0 = time.time()
        page_items_map = _build_all_page_items(doc)
        total_items = sum(len(v) for v in page_items_map.values())
        logging.info(f"[Row {idx}] Step 4a/6: Single-pass item bucketing done in {time.time()-t0:.2f}s ({total_items} items across {len(page_items_map)} pages)")

        # step 4b: build page metadata from buckets (threaded, no doc access)
        t0 = time.time()
        pages_content = [None] * num_pages
        args_list = [
            (page_no, page_items_map.get(page_no, []), count_tokens, tables_by_page.get(page_no, []))
            for page_no in range(num_pages)
        ]
        with ThreadPoolExecutor(max_workers=POST_PROCESS_WORKERS) as executor:
            futures = {
                executor.submit(_build_page_metadata, args): args[0]
                for args in args_list
            }
            done_count = 0
            for future in as_completed(futures):
                page_no = futures[future]
                try:
                    pages_content[page_no] = future.result()
                except Exception as e:
                    logging.error(f"[Row {idx}] Page {page_no} metadata build failed: {e}", exc_info=True)
                    pages_content[page_no] = {"page_no": page_no, "content": "", "error": str(e)}
                done_count += 1
                if done_count % 200 == 0 or done_count == num_pages:
                    logging.info(f"[Row {idx}] Step 4b/6: Page metadata progress {done_count}/{num_pages}")
        logging.info(f"[Row {idx}] Step 4b/6: Page metadata build done in {time.time()-t0:.2f}s ({num_pages} pages, {POST_PROCESS_WORKERS} threads)")

        in_refs = False
        for page in pages_content:
            if page is None:
                continue
            for item in page.get("items", []):
                if item["label"] == "section_header":
                    if any(t in item["text"].lower() for t in
                           ["references", "bibliography", "works cited", "literature cited"]):
                        in_refs = True
                item["is_reference"] = in_refs
            # recompute page-level fields
            if in_refs:
                ref_texts = []
                non_ref_texts = []
                for item in page.get("items", []):
                    if isinstance(item["text"], list):
                        text = _render_table_markdown(item["text"])
                    elif isinstance(item["text"], str):
                        text = item["text"]
                    else:
                        text = ""
                    if not text:
                        continue
                    if item["is_reference"]:
                        ref_texts.append(text)
                    else:
                        non_ref_texts.append(text)
                page["content_before_references"] = "\n\n".join(non_ref_texts)
                page["reference_content"] = "\n\n".join(ref_texts)
                page["has_references"] = len(ref_texts) > 0
                page["token_count_before_references"] = count_tokens(page["content_before_references"])
                page["token_count_references"] = count_tokens(page["reference_content"])

        # step 5: extract formulas with provenance
        t0 = time.time()
        formula_list = []
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel
        for el in doc.texts:
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA and el.text != "$$MALFORMED_FORMULA$$":
                formula_data = {"latex": el.text}
                if el.prov:
                    formula_data["provenance"] = [
                        {
                            "page_no": prov.page_no,
                            "bbox": {
                                "l": prov.bbox.l,
                                "t": prov.bbox.t,
                                "r": prov.bbox.r,
                                "b": prov.bbox.b
                            } if prov.bbox else None
                        }
                        for prov in el.prov
                    ]
                formula_list.append(formula_data)
        logging.info(f"[Row {idx}] Step 5/6: Formula extraction done in {time.time()-t0:.2f}s ({len(formula_list)} formulas)")

        token_count = count_tokens(text_md)

        total_elapsed = time.time() - t_total_start
        logging.info(f"[Row {idx}] Extraction complete — total {total_elapsed:.2f}s")

        return {
            "FullText": text_md,
            "PagesJson": json.dumps(pages_content, ensure_ascii=False),
            "TablesJson": json.dumps(all_tables_json, ensure_ascii=False),
            "EquationsJson": json.dumps(formula_list, ensure_ascii=False),
            "TokenCount": token_count,
            "NumPages": num_pages,
            "NumTables": num_tables,
            "NumPictures": num_pictures,
            "Error": None
        }

    except Exception as e:
        logging.error(f"[Row {idx}] Docling extraction failed for {pdf_path}: {e}", exc_info=True)
        return {
            "FullText": "ANALYSIS_ERROR",
            "PagesJson": "ANALYSIS_ERROR",
            "TablesJson": "ANALYSIS_ERROR",
            "EquationsJson": "ANALYSIS_ERROR",
            "TokenCount": 0,
            "NumPages": 0,
            "NumTables": 0,
            "NumPictures": 0,
            "Error": str(e)
        }

def _default_result(error: str | None = None) -> dict:
    return {
        "FullText": "ANALYSIS_ERROR" if error else "",
        "PagesJson": "ANALYSIS_ERROR" if error else "",
        "TablesJson": "ANALYSIS_ERROR" if error else "",
        "EquationsJson": "ANALYSIS_ERROR" if error else "",
        "TokenCount": 0,
        "NumPages": 0,
        "NumTables": 0,
        "NumPictures": 0,
        "Error": error,
    }


def initialize_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = _default_result()[col]
    return out


def validate_output_row(pdf_path: str, file_name: str, raw_result: dict) -> dict:
    try:
        row = ExtractionRow(PDFPath=pdf_path, FileName=file_name, **raw_result)
    except ValidationError as exc:
        row = ExtractionRow(PDFPath=pdf_path, FileName=file_name, **_default_result(f"VALIDATION_ERROR: {exc}"))
    return row.model_dump()


def assign_row_result(df: pd.DataFrame, row_index: Any, result: dict) -> None:
    for col in OUTPUT_COLUMNS:
        df.at[row_index, col] = result[col]


def row_is_complete(row: pd.Series) -> bool:
    return row.get("Error") is None and bool(row.get("FullText"))


def row_is_timeout(row: pd.Series) -> bool:
    return row.get("Error") == "TIMEOUT"


def row_should_retry(row: pd.Series, retry_errors: bool) -> bool:
    if row_is_complete(row):
        return False
    if row_is_timeout(row):
        return True
    error = row.get("Error")
    if error == "IN_PROGRESS":
        return True
    if error in (None, "", 0):
        return True
    return retry_errors


def load_or_initialize_dataframe(input_df: pd.DataFrame, output_path: Path | None, resume: bool, retry_errors: bool) -> pd.DataFrame:
    base = initialize_output_dataframe(input_df)
    if not resume or output_path is None or not output_path.exists():
        return base

    existing = pd.read_feather(output_path)
    missing = [col for col in OUTPUT_COLUMNS if col not in existing.columns]
    if missing:
        raise ValueError(f"Existing resume file is missing columns: {missing}")
    if len(existing) != len(base):
        raise ValueError("Existing resume file row count does not match input row count")
    if existing["PDFPath"].tolist() != base["PDFPath"].tolist() or existing["FileName"].tolist() != base["FileName"].tolist():
        raise ValueError("Existing resume file does not match input PDF ordering")

    for idx in base.index:
        row = existing.loc[idx]
        if row_is_complete(row) or row_should_retry(row, retry_errors):
            for col in RESULT_COLUMNS:
                base.at[idx, col] = row[col]
        elif row.get("Error"):
            for col in RESULT_COLUMNS:
                base.at[idx, col] = row[col]
    return base


def write_checkpoint(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    df.to_feather(tmp_path)
    os.replace(tmp_path, output_path)


def should_flush_checkpoint(dirty_count: int, last_save_time: float, now: float, save_every: int, save_interval_seconds: float) -> bool:
    if dirty_count <= 0:
        return False
    if dirty_count >= max(1, save_every):
        return True
    return (now - last_save_time) >= save_interval_seconds


def create_executor(max_workers: int):
    return ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer)


def shutdown_executor(executor, *, kill_workers: bool) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=not kill_workers, cancel_futures=True)
    finally:
        if kill_workers:
            processes = getattr(executor, "_processes", None) or {}
            for proc in processes.values():
                if proc.is_alive():
                    proc.terminate()
            for proc in processes.values():
                proc.join(timeout=5)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()


def pending_row_indices(df: pd.DataFrame, retry_errors: bool) -> list[Any]:
    return [idx for idx in df.index if row_should_retry(df.loc[idx], retry_errors)]


def mark_missing_pdf(df: pd.DataFrame, row_index: Any, pdf_path: str, file_name: str) -> None:
    result = validate_output_row(pdf_path, file_name, _default_result("PDF_NOT_FOUND"))
    assign_row_result(df, row_index, result)


def mark_timeout(df: pd.DataFrame, row_index: Any, pdf_path: str, file_name: str) -> None:
    result = validate_output_row(pdf_path, file_name, _default_result("TIMEOUT"))
    assign_row_result(df, row_index, result)


def mark_in_progress(df: pd.DataFrame, row_index: Any, pdf_path: str, file_name: str) -> None:
    result = validate_output_row(
        pdf_path,
        file_name,
        {
            "FullText": "",
            "PagesJson": "",
            "TablesJson": "",
            "EquationsJson": "",
            "TokenCount": 0,
            "NumPages": 0,
            "NumTables": 0,
            "NumPictures": 0,
            "Error": "IN_PROGRESS",
        },
    )
    assign_row_result(df, row_index, result)


def do_docling_extraction(
    df: pd.DataFrame,
    max_workers=10,
    output_dir=None,
    output_path: str | None = None,
    save_every: int = 1,
    save_interval_seconds: float = 30.0,
    task_timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 5.0,
    resume: bool = False,
    retry_errors: bool = False,
    executor_factory=create_executor,
    wait_fn=wait,
    clock_fn=time.monotonic,
) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using a restartable ProcessPoolExecutor.
    Completed rows are checkpointed incrementally; timed-out rows are marked and
    the executor is recreated so the run can continue.
    """
    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Multi-GPU with Provenance)...")

    checkpoint_path = Path(output_path) if output_path else None
    df = load_or_initialize_dataframe(df, checkpoint_path, resume=resume, retry_errors=retry_errors)
    dirty_count = 0
    last_save_time = clock_fn()

    rows_to_process = pending_row_indices(df, retry_errors=retry_errors)
    pending: list[Any] = []
    for idx in rows_to_process:
        pdf_path = df.at[idx, "PDFPath"]
        if not pdf_path or not os.path.exists(pdf_path):
            logging.warning(f"[MainProc] Missing PDF path for row {idx}: {pdf_path}")
            mark_missing_pdf(df, idx, str(pdf_path), str(df.at[idx, 'FileName']))
            dirty_count += 1
            now = clock_fn()
            if checkpoint_path and should_flush_checkpoint(dirty_count, last_save_time, now, save_every, save_interval_seconds):
                write_checkpoint(df, checkpoint_path)
                dirty_count = 0
                last_save_time = now
        else:
            pending.append(idx)

    executor = None
    in_flight: dict[Any, tuple[int, float]] = {}

    try:
        executor = executor_factory(max_workers)

        while pending or in_flight:
            while executor is not None and pending and len(in_flight) < max_workers:
                idx = pending.pop(0)
                pdf_path = df.at[idx, "PDFPath"]
                mark_in_progress(df, idx, str(pdf_path), str(df.at[idx, "FileName"]))
                dirty_count += 1
                logging.info(f"[MainProc] Submitting row {idx}, pdf={pdf_path}")
                future = executor.submit(extract_pdf_with_docling, pdf_path, idx, output_dir)
                in_flight[future] = (idx, clock_fn())

                now = clock_fn()
                if checkpoint_path and should_flush_checkpoint(dirty_count, last_save_time, now, save_every, save_interval_seconds):
                    write_checkpoint(df, checkpoint_path)
                    dirty_count = 0
                    last_save_time = now

            if not in_flight:
                break

            done, _ = wait_fn(set(in_flight.keys()), timeout=poll_interval_seconds, return_when=FIRST_COMPLETED)
            now = clock_fn()

            if not done:
                overdue = [future for future, (_, started_at) in in_flight.items() if now - started_at >= task_timeout_seconds]
                if overdue:
                    timed_out_rows = {in_flight[future][0] for future in overdue}
                    lost_rows = [idx for future, (idx, _) in in_flight.items() if future not in overdue]

                    for idx in sorted(timed_out_rows):
                        logging.error(f"[MainProc] Timeout row {idx} after {task_timeout_seconds}s")
                        mark_timeout(df, idx, str(df.at[idx, "PDFPath"]), str(df.at[idx, "FileName"]))
                        dirty_count += 1

                    if checkpoint_path:
                        write_checkpoint(df, checkpoint_path)
                        dirty_count = 0
                        last_save_time = now

                    shutdown_executor(executor, kill_workers=True)
                    executor = executor_factory(max_workers)
                    in_flight.clear()
                    pending = lost_rows + pending
                    continue

                if checkpoint_path and should_flush_checkpoint(dirty_count, last_save_time, now, save_every, save_interval_seconds):
                    write_checkpoint(df, checkpoint_path)
                    dirty_count = 0
                    last_save_time = now
                continue

            for future in done:
                idx, _ = in_flight.pop(future)
                pdf_path = str(df.at[idx, "PDFPath"])
                file_name = str(df.at[idx, "FileName"])
                try:
                    raw_result = future.result()
                except Exception as exc:
                    logging.exception(f"Multiprocessing extraction exception row {idx}: {exc}")
                    raw_result = _default_result(str(exc))

                result = validate_output_row(pdf_path, file_name, raw_result)
                assign_row_result(df, idx, result)
                dirty_count += 1

                if result["Error"]:
                    logging.error(f"[!] Extraction error row {idx}: {result['Error']}")
                    print(f"[!] Extraction error row {idx}: {result['Error']}")
                else:
                    logging.info(
                        "Extraction successful row %s, pages=%s, tables=%s, tokens=%s",
                        idx,
                        result["NumPages"],
                        result["NumTables"],
                        result["TokenCount"],
                    )
                    print(f"[OK] Extraction successful row {idx}: {result['NumPages']} pages, {result['NumTables']} tables")

            now = clock_fn()
            if checkpoint_path and should_flush_checkpoint(dirty_count, last_save_time, now, save_every, save_interval_seconds):
                write_checkpoint(df, checkpoint_path)
                dirty_count = 0
                last_save_time = now

    finally:
        shutdown_executor(executor, kill_workers=False)

    if checkpoint_path:
        write_checkpoint(df, checkpoint_path)
    logging.info("Multiprocessing Docling extraction complete.")
    return df


def create_pdf_dataframe(pdf_folder):
    """Scan a folder for PDF files and create a DataFrame for processing."""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder}")
    df = pd.DataFrame({
        "PDFPath": pdf_files,
        "FileName": [os.path.basename(p) for p in pdf_files]
    })
    return df


if __name__ == "__main__":
    from datetime import datetime

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE = os.path.join(LOG_DIR, f"multi_provenance_{timestamp}.log")

    logging.basicConfig(
        filename=LOG_FILE,
        filemode='w',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    PDF_FOLDER = os.path.join(SCRIPT_DIR, "sample_pdfs")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"multi_provenance_{timestamp}.feather")

    parser = argparse.ArgumentParser(description="Extract Docling outputs to feather with checkpointing and resume support.")
    parser.add_argument("pdf_folder", nargs="?", default=PDF_FOLDER, help="Folder containing PDFs to process")
    parser.add_argument("output_path", nargs="?", default=None, help="Destination feather path")
    parser.add_argument("--max-workers", type=int, default=len(GPU_IDS) * 5, help="Maximum number of worker processes")
    parser.add_argument("--task-timeout-seconds", type=float, default=600.0, help="Per-task timeout before executor restart")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0, help="Supervisor poll interval")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint after this many row updates")
    parser.add_argument("--save-interval-seconds", type=float, default=30.0, help="Checkpoint if this much time passes with unsaved progress")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output feather if present")
    parser.add_argument("--retry-errors", action="store_true", help="When resuming, retry rows with non-timeout errors")
    args = parser.parse_args()

    PDF_FOLDER = args.pdf_folder
    OUTPUT_PATH = args.output_path or DEFAULT_OUTPUT_PATH

    logging.info(f"PDF folder: {PDF_FOLDER}")
    logging.info(f"Output: {OUTPUT_PATH}")
    logging.info(f"Log: {LOG_FILE}")

    df = create_pdf_dataframe(PDF_FOLDER)
    logging.info(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(df[["PDFPath", "FileName"]].to_string())

    df = do_docling_extraction(
        df,
        max_workers=args.max_workers,
        output_dir=OUTPUT_DIR,
        output_path=OUTPUT_PATH,
        save_every=args.save_every,
        save_interval_seconds=args.save_interval_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )

    logging.info(f"Saved results to {OUTPUT_PATH}")
    print(f"Saved results to {OUTPUT_PATH}")
