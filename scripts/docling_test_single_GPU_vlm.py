# -*- coding: utf-8 -*-
"""
Multiprocessing Docling VLM extraction using VLLM API endpoint

This version uses the VLLM API server running on port 8006 (granite-vision-3.3-2b)
instead of loading the model in each worker process.
- All workers send requests to http://localhost:8006
- No GPU assignment needed per worker (API server handles GPU)
- Much more memory efficient - only one model instance running
"""
import logging
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

# no need to set CUDA variables - API server handles GPU
# multiprocessing.set_start_method("spawn", force=True)

def worker_initializer():
    """Initialize worker process with logging only - no GPU setup needed"""
    try:
        import logging
        import sys
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

        # also add stderr handler so we can see errors
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(stderr_handler)

        logging.info(f"Worker {multiprocessing.current_process().name} initialized for API-based extraction")
    except Exception as e:
        import sys
        print(f"WORKER INIT FAILED: {e}", file=sys.stderr)
        raise


def init_tokenizer():
    """Initialize tokenizer for counting tokens"""
    try:
        import tiktoken
        tok = tiktoken.get_encoding("gpt2")
        return lambda text: len(tok.encode(text)) if text else 0
    except Exception:
        return lambda text: 0


def extract_pdf_with_docling(pdf_path):
    """
    Uses DocumentConverter with VLM pipeline connecting to VLLM API endpoint.
    Each worker sends requests to http://localhost:8006 instead of loading the model.
    """
    logging.info(f"[Worker {multiprocessing.current_process().name}] Starting extraction for {pdf_path}")

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel
        from pydantic import AnyUrl

        # initialize tokenizer
        count_tokens = init_tokenizer()

        # create API VLM options pointing to VLLM server on port 8006
        logging.info(f"[Worker {multiprocessing.current_process().name}] Configuring API VLM options for localhost:8006...")

        vlm_options = ApiVlmOptions(
            url=AnyUrl("http://localhost:8006/v1/chat/completions"),
            params={"model": "ibm-granite/granite-vision-3.3-2b"},
            prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
            scale=2.0,
            timeout=120,
            response_format=ResponseFormat.MARKDOWN,
            temperature=0.0,
            concurrency=1,  # Each worker processes one page at a time
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            generate_page_images=True,
            force_backend_text=False,
            enable_remote_services=True,  # Required for API-based VLM
        )

        # create format option for VLM
        format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options
        )

        # initialize DocumentConverter with VLM pipeline
        logging.info(f"[Worker {multiprocessing.current_process().name}] Initializing DocumentConverter with API VLM...")
        converter = DocumentConverter(
            format_options={InputFormat.PDF: format_option}
        )
        logging.info(f"[Worker {multiprocessing.current_process().name}] DocumentConverter initialized")

        # convert PDF using DocumentConverter
        logging.info(f"[Worker {multiprocessing.current_process().name}] Converting PDF via API...")
        conv_res = converter.convert(Path(pdf_path))
        logging.info(f"[Worker {multiprocessing.current_process().name}] Conversion complete")

        doc = conv_res.document

        # extract document metadata
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)

        # export to markdown
        text_md = doc.export_to_markdown()

        # clean artifacts
        artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
        for artifact in artifacts_to_remove:
            text_md = text_md.replace(artifact, "")

        # extract formulas
        formula_list = [
            {"latex": el.text}
            for el in doc.texts
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA and el.text != "$$MALFORMED_FORMULA$$"
        ]

        # process tables
        import warnings
        all_tables_json = []
        for table in doc.tables:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="DataFrame columns are not unique")
                all_tables_json.append(table.export_to_dataframe(doc=doc).to_dict(orient="records"))

        token_count = count_tokens(text_md)

        logging.info(f"[Worker {multiprocessing.current_process().name}] Extraction successful - {num_pages} pages, {token_count} tokens")
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
        logging.error(f"Docling VLM extraction failed for {pdf_path}: {e}", exc_info=True)
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


def do_docling_extraction(df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    """
    Multiprocessing extraction using API endpoint.
    All workers send requests to the same VLLM server on port 8006.
    """
    logging.info(f"Starting multiprocessing API-based VLM extraction for {len(df)} records using {max_workers} worker(s)")
    print("[Step 9/11] Extracting text/tables/formulas via Docling VLM API (Multiprocessing)...")
    print(f"[INFO] Using VLLM API endpoint at http://localhost:8006")

    # ensure output columns
    for col in ["FullText", "TablesJson", "EquationsJson", "TokenCount", "NumPages", "NumTables", "NumPictures", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else (0 if col.startswith("Num") else "")

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        logging.info(f"ProcessPoolExecutor created with {max_workers} workers")
        for idx, row in df.iterrows():
            pdf_path = row.get("PDFPath", "")
            if pdf_path and os.path.exists(pdf_path):
                future = executor.submit(extract_pdf_with_docling, pdf_path)
                futures[future] = idx
                logging.info(f"[MainProc] Submitted row {idx}, pdf={pdf_path}")
                print(f" -> Submitted PDF row {idx}, file: {os.path.basename(pdf_path)}")
            else:
                logging.warning(f"Row {idx} skipped: PDF path not found or empty.")
                df.at[idx, "Error"] = "PDF_NOT_FOUND"
                df.at[idx, "FullText"] = "ANALYSIS_ERROR"

        logging.info(f"All {len(futures)} tasks submitted. Waiting for completion...")
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
                    print(f"[OK] Extraction successful row {irow}: {result['NumPages']} pages, {result['NumTables']} tables, {result['TokenCount']} tokens")

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

    logging.info("Multiprocessing API-based VLM Docling extraction complete.")
    return df
