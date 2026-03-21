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
import glob
# import any other modules that are GPU related only after the workers are initialized

# single GPU ID to use:
GPU_ID = 4

# debug output path for layout visualization images
# set this before calling do_docling_extraction() to override the default
# default: <script_dir>/../docling_debug
DEBUG_OUTPUT_PATH = None

# per-worker globals (initialized once in worker_initializer, reused across PDFs)
_worker_converter = None
_worker_tokenizer = None

def worker_initializer():
    """
    This initializer is called once per worker process.
    It assigns all workers to the single specified GPU,
    then loads the converter and tokenizer once for reuse.
    """
    global _worker_converter, _worker_tokenizer

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

    # load converter and tokenizer once per worker
    _worker_converter = init_debug_pipeline()
    _worker_tokenizer = init_tokenizer()
    logging.info(f"Worker {multiprocessing.current_process().name} converter and tokenizer initialized")

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
    pipeline_options.generate_page_images = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.images_scale = 2.0

    # enable debug visualization for bounding boxes
    debug_path = DEBUG_OUTPUT_PATH
    if debug_path is None:
        debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docling_debug")
    os.makedirs(debug_path, exist_ok=True)
    settings.debug.visualize_raw_layout = True
    settings.debug.visualize_layout = True
    settings.debug.debug_output_path = debug_path

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
    In the child process, reinitialize the debug pipeline, convert the PDF,
    and return the results with page-level provenance.
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

def do_docling_extraction(df: pd.DataFrame, max_workers=5, output_dir=None) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using the ProcessPoolExecutor.
    This version uses a single GPU for all workers.
    """
    global DEBUG_OUTPUT_PATH
    if output_dir is not None:
        DEBUG_OUTPUT_PATH = output_dir
    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Single-GPU Debug with Provenance)...")
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
    import sys
    from datetime import datetime

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE = os.path.join(LOG_DIR, f"single_debug_provenance_{timestamp}.log")

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
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"single_debug_provenance_{timestamp}.feather")

    if len(sys.argv) > 1:
        PDF_FOLDER = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        DEBUG_OUTPUT_PATH = sys.argv[3]

    logging.info(f"PDF folder: {PDF_FOLDER}")
    logging.info(f"Output: {OUTPUT_PATH}")
    logging.info(f"Debug images: {DEBUG_OUTPUT_PATH or os.path.join(SCRIPT_DIR, '..', 'docling_debug')}")
    logging.info(f"Log: {LOG_FILE}")

    df = create_pdf_dataframe(PDF_FOLDER)
    logging.info(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(df[["PDFPath", "FileName"]].to_string())

    df = do_docling_extraction(df, max_workers=5)

    df.to_feather(OUTPUT_PATH)
    logging.info(f"Saved results to {OUTPUT_PATH}")
    print(f"Saved results to {OUTPUT_PATH}")
