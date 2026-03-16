# docling_extract_formulas_mp_singl.py

import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
import logging
import json
import tiktoken
import glob
# import any other modules that are GPU related only after the workers are initialized

# single GPU ID to use:
GPU_ID = 4

# debug output path for layout visualization images
# set this before calling do_docling_extraction() to enable debug mode
# when None, debug visualization is disabled
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
    Called once in each worker process to create a DocumentConverter.
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

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA
    )

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

    # enable debug visualization if DEBUG_OUTPUT_PATH is set
    if DEBUG_OUTPUT_PATH is not None:
        from docling.datamodel.settings import settings
        os.makedirs(DEBUG_OUTPUT_PATH, exist_ok=True)
        settings.debug.visualize_raw_layout = True
        settings.debug.visualize_layout = True
        settings.debug.debug_output_path = DEBUG_OUTPUT_PATH

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    return converter


def _export_single_table(table, doc):
    """Worker function for threaded table export."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrame columns are not unique")
        return table.export_to_dataframe(doc=doc).to_dict(orient="records")


# number of threads for post-processing (tables)
POST_PROCESS_WORKERS = 12


def extract_pdf_with_docling(pdf_path: str, idx: int, output_dir=None) -> dict:
    """
    In the child process, convert the PDF and return results.
    No page-level provenance or reference tracking.
    """
    global _worker_converter, _worker_tokenizer
    t_total_start = time.time()
    logging.info(f"[Row {idx}] Starting extraction for {pdf_path}")

    converter = _worker_converter
    count_tokens = _worker_tokenizer

    try:
        # step 1: convert document (GPU-accelerated)
        t0 = time.time()
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)
        logging.info(f"[Row {idx}] Step 1/4: Docling conversion done in {time.time()-t0:.2f}s — {num_pages} pages, {num_tables} tables, {num_pictures} pictures")

        # step 2: full markdown export
        t0 = time.time()
        text_md = doc.export_to_markdown()
        for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
            text_md = text_md.replace(artifact, "")
        logging.info(f"[Row {idx}] Step 2/4: Full markdown export done in {time.time()-t0:.2f}s ({len(text_md)} chars)")

        # step 3: export tables (threaded)
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
                        table_results[t_idx] = []
                all_tables_json = table_results
        logging.info(f"[Row {idx}] Step 3/4: Table export done in {time.time()-t0:.2f}s ({num_tables} tables)")

        # step 4: extract formulas
        t0 = time.time()
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel
        formula_list = [
            {"latex": el.text}
            for el in doc.texts
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA and el.text != "$$MALFORMED_FORMULA$$"
        ]
        logging.info(f"[Row {idx}] Step 4/4: Formula extraction done in {time.time()-t0:.2f}s ({len(formula_list)} formulas)")

        token_count = count_tokens(text_md)

        total_elapsed = time.time() - t_total_start
        logging.info(f"[Row {idx}] Extraction complete — total {total_elapsed:.2f}s")

        return {
            "FullText": text_md,
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
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Single-GPU)...")
    for col in ["FullText", "TablesJson", "EquationsJson", "TokenCount", "NumPages", "NumTables", "NumPictures", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else (0 if col.startswith("Num") else "")

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        for idx, row in df.iterrows():
            pdf_path = row.get("PDFPath", "")
            if not pdf_path or not os.path.exists(pdf_path):
                logging.warning(f"[MainProc] Missing PDF path for row {idx}: {pdf_path}")
                df.at[idx, "FullText"] = "ANALYSIS_ERROR"
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
    LOG_FILE = os.path.join(LOG_DIR, f"singl_{timestamp}.log")

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
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"singl_{timestamp}.feather")

    if len(sys.argv) > 1:
        PDF_FOLDER = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        DEBUG_OUTPUT_PATH = sys.argv[3]

    logging.info(f"PDF folder: {PDF_FOLDER}")
    logging.info(f"Output: {OUTPUT_PATH}")
    logging.info(f"Debug images: {DEBUG_OUTPUT_PATH or 'disabled'}")
    logging.info(f"Log: {LOG_FILE}")

    df = create_pdf_dataframe(PDF_FOLDER)
    logging.info(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(f"Found {len(df)} PDFs in {PDF_FOLDER}")
    print(df[["PDFPath", "FileName"]].to_string())

    df = do_docling_extraction(df, max_workers=5)

    df.to_feather(OUTPUT_PATH)
    logging.info(f"Saved results to {OUTPUT_PATH}")
    print(f"Saved results to {OUTPUT_PATH}")
