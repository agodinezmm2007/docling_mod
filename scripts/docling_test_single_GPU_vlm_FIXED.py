# -*- coding: utf-8 -*-
"""
Multiprocessing single-GPU Docling VLM extraction - PROPERLY USING DocumentConverter

This version uses Vision-Language Models (VLM) for end-to-end PDF conversion:
- Uses VlmPipeline via DocumentConverter (the correct way)
- Converts entire PDF pages to markdown using Granite Vision (2B) model
- VLM directly processes page images without layout detection/OCR steps
"""
import logging
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

# ensure fresh spawn start to avoid inheriting any CUDA state
multiprocessing.set_start_method("spawn", force=True)

# list of GPU IDs you want to use (single-GPU)
GPU_IDS = ["1"]

# global variables for assigning GPUs
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)

def worker_initializer():
    try:
        with assign_lock, next_gpu.get_lock():
            gpu_index = next_gpu.value % len(GPU_IDS)
            next_gpu.value += 1

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS[gpu_index]
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32768"

        # configure logging for worker process
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

        logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")
        logging.info(f"Worker {multiprocessing.current_process().name} initialization complete")
    except Exception as e:
        import sys
        print(f"WORKER INIT FAILED: {e}", file=sys.stderr)
        raise


def init_tokenizer():
    try:
        import tiktoken
        tok = tiktoken.get_encoding("gpt2")
        return lambda text: len(tok.encode(text)) if text else 0
    except Exception:
        return lambda text: 0


def extract_pdf_with_docling(pdf_path):
    """
    Uses DocumentConverter with VLM pipeline - the CORRECT way to use Docling VLM.
    """
    logging.info(f"[Worker {multiprocessing.current_process().name}] Starting extraction for {pdf_path}")

    try:
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.vlm_model_specs import GRANITE_VISION_TRANSFORMERS
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel

        # initialize tokenizer
        count_tokens = init_tokenizer()

        # create VLM pipeline options
        logging.info(f"[Worker {multiprocessing.current_process().name}] Creating VLM pipeline options...")
        accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA
        )

        pipeline_options = VlmPipelineOptions(
            accelerator_options=accelerator_options,
            vlm_options=GRANITE_VISION_TRANSFORMERS,
            generate_page_images=True,
            force_backend_text=False,
        )

        # create format option for VLM
        format_option = FormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options
        )

        # initialize DocumentConverter with VLM pipeline
        logging.info(f"[Worker {multiprocessing.current_process().name}] Initializing DocumentConverter with VLM...")
        converter = DocumentConverter(
            format_options={InputFormat.PDF: format_option}
        )
        logging.info(f"[Worker {multiprocessing.current_process().name}] DocumentConverter initialized")

        # convert PDF using DocumentConverter (the correct way)
        logging.info(f"[Worker {multiprocessing.current_process().name}] Converting PDF...")
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

        logging.info(f"[Worker {multiprocessing.current_process().name}] Extraction successful")
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
    Multiprocessing extraction using a worker initializer to pin each process to a GPU.
    """
    logging.info(f"Starting multiprocessing Docling extraction for {len(df)} records using {max_workers} worker(s)")
    print("[Step 9/11] Extracting text/tables/formulas via Docling VLM (Multiprocessing)...")

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

    logging.info("Multiprocessing Docling extraction complete.")
    return df
