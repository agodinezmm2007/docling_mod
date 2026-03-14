# docling_extract_formulas_mp_multi.py (Multiprocessing Version with Dual GPU)

import os
import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import logging
import json
import tiktoken

# list of GPU IDs you want to use:
GPU_IDS = [1, 2]

# global variables for assigning GPUs
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)

def worker_initializer():
    """
    This initializer is called once per worker process.
    It uses a global counter to assign one GPU from GPU_IDS to each worker.
    """
    with assign_lock, next_gpu.get_lock():
        # determine which GPU to assign based on a round-robin strategy
        gpu_index = next_gpu.value % len(GPU_IDS)
        next_gpu.value += 1
    # iMPORTANT: Set CUDA_VISIBLE_DEVICES early!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_index])

    # set other CUDA-related env vars as need
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32768"

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

def init_converter():
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

    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )

    # configure layout model - Using HERON_101 (76.7M params)
    layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON_101
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.layout_options = layout_options
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_table_structure = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    return converter 


def extract_pdf_with_docling(pdf_path):
    """
    In the child process, reinitialize the converter pipeline, convert the PDF,
    parse formulas, and return the results.
    """
    converter = init_converter()
    count_tokens = init_tokenizer()

    try:
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        # extract document metadata
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)

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
        import warnings
        all_tables_json = []
        for table in doc.tables:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="DataFrame columns are not unique")
                all_tables_json.append(table.export_to_dataframe(doc=doc).to_dict(orient="records"))

        token_count = count_tokens(text_md)

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
        logging.error(f"Docling extraction failed for {pdf_path}: {e}", exc_info=True)
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

def do_docling_extraction(df: pd.DataFrame, max_workers=14) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using the ProcessPoolExecutor.
    This version uses the worker_initializer to distribute GPUs.
    """

    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Dual GPU)...")
    for col in ["FullText", "TablesJson", "EquationsJson", "TokenCount", "NumPages", "NumTables", "NumPictures", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else (0 if col.startswith("Num") else "")

    futures = {}
    # the initializer ensures each worker gets its GPU assigned
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
            future = executor.submit(extract_pdf_with_docling, pdf_path)
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
