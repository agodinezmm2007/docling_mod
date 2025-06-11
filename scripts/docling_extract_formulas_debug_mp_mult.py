# docling_extract_formulas_debug_mp_multi.py

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import logging
import json
import tiktoken
# import any other modules that are GPU related only after the workers are initialized

# List of GPU IDs you want to use:
GPU_IDS = [1, 2]

# Global variables for assigning GPUs
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)

def worker_initializer():
    """
    This initializer is called once per worker process.
    It uses a global counter to assign one GPU from GPU_IDS to each worker.
    """
    with assign_lock, next_gpu.get_lock():
        # Determine which GPU to assign based on a round-robin strategy
        gpu_index = next_gpu.value % len(GPU_IDS)
        next_gpu.value += 1
    # IMPORTANT: Set CUDA_VISIBLE_DEVICES early!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_index])
    # set other CUDA-related env vars as needed
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32768"
    # Log to see which worker got which GPU
    logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import settings

    # adjust additional environment settings if required
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.images_scale = 2.0

    # Enable debug visualization for bounding boxes
    settings.debug.visualize_raw_layout = True
    settings.debug.visualize_layout = True
    settings.debug.debug_output_path = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/docling_debug"

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    return converter

def extract_pdf_with_docling(pdf_path: str, idx: int, output_dir=None) -> dict:
    """
    In the child process, reinitialize the debug pipeline, convert the PDF,
    parse formulas, and return the results.
    """
    count_tokens = init_tokenizer()
    converter = init_debug_pipeline()

    try:
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document

        text_md = doc.export_to_markdown()

        # --- NEW CLEANING STEP ---
        artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
        for artifact in artifacts_to_remove:
            text_md = text_md.replace(artifact, "")
        # -------------------------

        # get formulas from the document
        formula_list = []
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel
        for el in doc.texts:
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA and el.text != "$$MALFORMED_FORMULA$$":
                formula_list.append({"latex": el.text})

        # process tables
        all_tables_json = []
        for table in doc.tables:
            df = table.export_to_dataframe()
            table_records = df.to_dict(orient="records")
            all_tables_json.append(table_records)

        token_count = count_tokens(text_md)

        return {
            "FullText": text_md,
            "TablesJson": json.dumps(all_tables_json, ensure_ascii=False),
            "EquationsJson": json.dumps(formula_list, ensure_ascii=False),
            "TokenCount": token_count,
            "Error": None
        }

    except Exception as e:
        logging.error(f"[Worker] Error processing row {idx}, pdf={pdf_path}: {e}", exc_info=True)
        return {
            "FullText": "ANALYSIS_ERROR",
            "TablesJson": "ANALYSIS_ERROR",
            "EquationsJson": "ANALYSIS_ERROR",
            "TokenCount": 0,
            "Error": str(e)
        }

def do_docling_extraction(df: pd.DataFrame, max_workers=5, output_dir=None) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using the ProcessPoolExecutor.
    This version uses the worker_initializer to distribute GPUs.
    """
    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    for col in ["FullText", "TablesJson", "EquationsJson", "TokenCount", "Error"]:
        if col not in df.columns:
            df[col] = "" if col != "Error" else None

    futures = {}
    # The initializer ensures each worker gets its GPU assigned
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
                df.at[irow, "Error"] = result["Error"]

                if result["Error"]:
                    logging.error(f"[MainProc] Row {irow} extraction error: {result['Error']}")
                else:
                    logging.info(f"[MainProc] Row {irow} processed, tokens={result['TokenCount']}")
            except Exception as e:
                logging.exception(f"[MainProc] Future exception for row {irow}: {e}")
                df.at[irow, "FullText"] = "ANALYSIS_ERROR"
                df.at[irow, "TablesJson"] = "ANALYSIS_ERROR"
                df.at[irow, "EquationsJson"] = "ANALYSIS_ERROR"
                df.at[irow, "TokenCount"] = 0
                df.at[irow, "Error"] = str(e)

    logging.info("Docling extraction complete.")
    return df


# if __name__ == "__main__":
#     import sys
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )

#     TEST_PATH = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/docling_test_30.feather"
#     OUTPUT_PATH = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/enriched_output_debug_mproc.feather"

#     if len(sys.argv) > 1:
#         TEST_PATH = sys.argv[1]
#     if len(sys.argv) > 2:
#         OUTPUT_PATH = sys.argv[2]

#     df = pd.read_feather(TEST_PATH)
#     logging.info(f"Loaded {len(df)} rows from {TEST_PATH}")

#     # Set max_workers to 2 if you have 2 dedicated GPUs
#     df = do_docling_extraction(df, max_workers=2)

#     df.to_feather(OUTPUT_PATH)
#     logging.info(f"Saved results to {OUTPUT_PATH}")
