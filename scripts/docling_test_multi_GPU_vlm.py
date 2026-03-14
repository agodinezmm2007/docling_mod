
"""
Multiprocessing Multi-GPU Docling VLM extraction with worker initializer

This version uses Vision-Language Models (VLM) for end-to-end PDF conversion:
- Uses VlmPipeline instead of StandardPdfPipeline
- Converts entire PDF pages to markdown using Granite Vision (2B) model
- VLM directly processes page images without layout detection/OCR steps
- Better for complex layouts, figures, and context-aware extraction
- Slower but more accurate than standard pipeline
- Distributes work across multiple GPUs for parallel processing
"""
import logging
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# ensure fresh spawn start to avoid inheriting any CUDA state
# multiprocessing.set_start_method("spawn", force=True)

# list of GPU IDs you want to use (multi-GPU)
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

    # important: Set CUDA_VISIBLE_DEVICES early!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_index])

    # set other CUDA-related env vars as needed
    # VLM may need more memory than standard pipeline
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


def init_converter():
    """
    Called once in each worker process to create a VLM pipeline.
    Note: Since we already set CUDA_VISIBLE_DEVICES in the initializer,
    we do not need to reset it here.
    """
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.vlm_model_specs import GRANITE_VISION_TRANSFORMERS

    accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CUDA
    )

    # use Granite Vision VLM model for full page conversion
    pipeline_options = VlmPipelineOptions(
        accelerator_options=accelerator_options,
        vlm_options=GRANITE_VISION_TRANSFORMERS,
        generate_page_images=True,  # Required for VLM
        force_backend_text=False,   # False = use VLM-generated text
    )

    return VlmPipeline(pipeline_options=pipeline_options)


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
    In the child process, reinitialize the VLM pipeline, convert the PDF,
    parse formulas, and return the results.
    """
    pipeline = init_converter()
    count_tokens = init_tokenizer()

    from pathlib import Path
    from docling.datamodel.document import InputDocument
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import TextItem
    from docling_core.types.doc.labels import DocItemLabel

    try:
        # create input document for VLM pipeline
        input_doc = InputDocument(
            path_or_stream=Path(pdf_path),
            format=InputFormat.PDF,
            backend=PyPdfiumDocumentBackend
        )

        # execute VLM pipeline
        conv_res = pipeline.execute(input_doc, raises_on_error=False)
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


def do_docling_extraction(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """
    Processes each PDF row in parallel using the ProcessPoolExecutor.
    This version uses the worker_initializer to distribute GPUs across workers.
    """
    logging.info("Starting multiprocessing VLM docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling VLM (Multiprocessing Multi-GPU)...")

    # ensure output columns exist
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
            print(f" -> Submitted PDF row {idx}, file: {os.path.basename(pdf_path)}")
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

    logging.info("Multiprocessing VLM Docling extraction complete.")
    return df
