# docling_extract_formulas_debug_mp_singl_provenance.py

import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
import logging
import json
import tiktoken
import re
# import any other modules that are GPU related only after the workers are initialized

# single GPU ID to use:
GPU_ID = 4

def worker_initializer():
    """
    This initializer is called once per worker process.
    It assigns all workers to the single specified GPU.
    """
    # IMPORTANT: set CUDA_VISIBLE_DEVICES early!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
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
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)

    logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")

def init_tokenizer():
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        return lambda text: len(tokenizer.encode(text)) if text else 0
    except ImportError:
        return lambda text: 0

def init_debug_pipeline():
    """
    Called once in each worker process to set environment variables and
    create a DocumentConverter with the debug pipeline.
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
    from docling.datamodel.settings import settings

    # adjust additional environment settings if required
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    accelerator_options = AcceleratorOptions(
        num_threads=22,
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
    pipeline_options.do_formula_enrichment = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_parsed_pages = True
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
    Buckets items by page_no, avoiding 803 separate full-tree scans.
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
                "is_reference": False  # Set during per-page post-processing
            }

            if page_no not in page_items_map:
                page_items_map[page_no] = []
            page_items_map[page_no].append(item_data)
            break  # Only take first prov per item

    return page_items_map


def _build_page_metadata(args):
    """
    Build page metadata from pre-bucketed items. No doc access needed.
    Safe for ThreadPoolExecutor since it only operates on plain dicts/strings.
    """
    page_no, page_items, count_tokens = args

    # detect references section (same logic as before, per-page)
    in_references = False
    for item in page_items:
        if item["label"] == "section_header":
            text_lower = item["text"].lower().strip()
            if any(term in text_lower for term in
                   ['references', 'bibliography', 'works cited', 'literature cited']):
                in_references = True
        item["is_reference"] = in_references

    # build page content from item texts
    all_texts = [item["text"] for item in page_items if item["text"]]
    page_md = "\n\n".join(all_texts)

    # clean artifacts
    for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
        page_md = page_md.replace(artifact, "")

    # separate content before and after references
    content_before_refs = [item["text"] for item in page_items
                           if not item["is_reference"] and item["text"]]
    reference_content = [item["text"] for item in page_items
                         if item["is_reference"] and item["text"]]

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
POST_PROCESS_WORKERS = 22


def extract_pdf_with_docling(pdf_path: str, idx: int, output_dir=None) -> dict:
    """
    In the child process, reinitialize the debug pipeline, convert the PDF,
    and return the results with page-level provenance.
    Post-processing uses ThreadPoolExecutor for parallelism within a single PDF.
    """
    t_total_start = time.time()
    logging.info(f"[Row {idx}] Starting extraction for {pdf_path}")

    # step 1: init tokenizer
    t0 = time.time()
    count_tokens = init_tokenizer()
    logging.info(f"[Row {idx}] Step 1/6: Tokenizer initialized in {time.time()-t0:.2f}s")

    # step 2: init pipeline
    t0 = time.time()
    converter = init_debug_pipeline()
    logging.info(f"[Row {idx}] Step 2/6: Pipeline initialized in {time.time()-t0:.2f}s")

    try:
        # step 3: convert document (GPU-accelerated)
        t0 = time.time()
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)
        logging.info(f"[Row {idx}] Step 3/6: Docling conversion done in {time.time()-t0:.2f}s — {num_pages} pages, {num_tables} tables, {num_pictures} pictures")

        # step 4: full markdown export
        t0 = time.time()
        text_md = doc.export_to_markdown()
        artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
        for artifact in artifacts_to_remove:
            text_md = text_md.replace(artifact, "")
        logging.info(f"[Row {idx}] Step 4/6: Full markdown export done in {time.time()-t0:.2f}s ({len(text_md)} chars)")

        # step 5a: single-pass iterate_items -> bucket by page (one tree traversal)
        t0 = time.time()
        page_items_map = _build_all_page_items(doc)
        total_items = sum(len(v) for v in page_items_map.values())
        logging.info(f"[Row {idx}] Step 5a/6: Single-pass item bucketing done in {time.time()-t0:.2f}s ({total_items} items across {len(page_items_map)} pages)")

        # step 5b: build page metadata from buckets (threaded, no doc access)
        t0 = time.time()
        pages_content = [None] * num_pages
        args_list = [
            (page_no, page_items_map.get(page_no, []), count_tokens)
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
                    logging.info(f"[Row {idx}] Step 5b/6: Page metadata progress {done_count}/{num_pages}")
        logging.info(f"[Row {idx}] Step 5b/6: Page metadata build done in {time.time()-t0:.2f}s ({num_pages} pages, {POST_PROCESS_WORKERS} threads)")

        # formula enrichment disabled
        formula_list = []

        # step 6: process tables with provenance (threaded)
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
        logging.info(f"[Row {idx}] Step 6/6: Table export done in {time.time()-t0:.2f}s ({num_tables} tables)")

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

def do_docling_extraction(df: pd.DataFrame, max_workers=14, output_dir=None) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using the ProcessPoolExecutor.
    This version uses a single GPU for all workers.
    """
    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Single GPU Debug with Provenance)...")
    for col in ["FullText", "PagesJson", "TablesJson", "EquationsJson", "TokenCount", "NumPages", "NumTables", "NumPictures", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else (0 if col.startswith("Num") else "")

    futures = {}
    # the initializer ensures each worker gets the single GPU assigned
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        for idx, row in df.iterrows():
            pdf_path = row.get("PDFPath", "")
            if not pdf_path or not os.path.exists(pdf_path):
                logging.warning(f"[MainProc] Missing PDF path for row {idx}: {pdf_path}")
                df.at[idx, "FullText"] = "ANALYSIS_ERROR"
                df.at[idx, "PagesJson"] = "ANALYSIS_ERROR"
                df.at[idx, "TablesJson"] = "ANALYSIS_ERROR"
                df.at[idx, "EquationsJson"] = "ANALYSIS_ERROR"
                df.at[idx, "TokenCount"] = 0
                df.at[idx, "Error"] = "PDF_NOT_FOUND"
                continue
            logging.info(f"[MainProc] Submitting row {idx}, pdf={pdf_path}")
            future = executor.submit(extract_pdf_with_docling, pdf_path, idx, output_dir)
            futures[future] = idx

        for future in as_completed(futures):
            irow = futures[future]
            try:
                result = future.result()
                df.at[irow, "FullText"] = result["FullText"]
                df.at[irow, "PagesJson"] = result["PagesJson"]
                df.at[irow, "TablesJson"] = result["TablesJson"]
                df.at[irow, "EquationsJson"] = result["EquationsJson"]
                df.at[irow, "TokenCount"] = result["TokenCount"]
                df.at[irow, "NumPages"] = result["NumPages"]
                df.at[irow, "NumTables"] = result["NumTables"]
                df.at[irow, "NumPictures"] = result["NumPictures"]
                df.at[irow, "Error"] = result["Error"]

                if result["Error"]:
                    logging.error(f"[!] Extraction error row {irow}: {result['Error']}")
                    print(f"[!] Extraction error row {irow}: {result['Error']}")
                else:
                    logging.info(f"Extraction successful row {irow}, pages={result['NumPages']}, tables={result['NumTables']}, tokens={result['TokenCount']}")
                    print(f"[OK] Extraction successful row {irow}: {result['NumPages']} pages, {result['NumTables']} tables")

            except Exception as e:
                logging.exception(f"Multiprocessing extraction exception row {irow}: {e}")
                df.at[irow, "FullText"] = "ANALYSIS_ERROR"
                df.at[irow, "PagesJson"] = "ANALYSIS_ERROR"
                df.at[irow, "TablesJson"] = "ANALYSIS_ERROR"
                df.at[irow, "EquationsJson"] = "ANALYSIS_ERROR"
                df.at[irow, "TokenCount"] = 0
                df.at[irow, "NumPages"] = 0
                df.at[irow, "NumTables"] = 0
                df.at[irow, "NumPictures"] = 0
                df.at[irow, "Error"] = str(e)

    logging.info("Multiprocessing Docling extraction complete.")
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    INPUT_PATH = "/mnt/c/Users/WSTATION/Desktop/docling_mods/scripts/sample_pdfs/pdf_df.feather"
    OUTPUT_PATH = "/mnt/c/Users/WSTATION/Desktop/docling_mods/scripts/sample_pdfs/pdf_df_output.feather"

    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]

    df = pd.read_feather(INPUT_PATH)
    logging.info(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(df[["PDFPath", "FileName"]].to_string())

    df = do_docling_extraction(df, max_workers=12)

    df.to_feather(OUTPUT_PATH)
    logging.info(f"Saved results to {OUTPUT_PATH}")
    print(f"Saved results to {OUTPUT_PATH}")
