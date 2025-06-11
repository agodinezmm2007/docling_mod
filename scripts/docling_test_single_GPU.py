# -*- coding: utf-8 -*-
"""
Multiprocessing single-GPU Docling extraction with worker initializer
"""
import logging
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Ensure fresh spawn start to avoid inheriting any CUDA state
multiprocessing.set_start_method("spawn", force=True)

# List of GPU IDs you want to use (single-GPU)
GPU_IDS = ["0"]

# Global variables for assigning GPUs
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)

def worker_initializer():
    """
    This initializer is called once per worker process.
    It assigns a GPU from GPU_IDS to the process in a round-robin fashion.
    """
    with assign_lock, next_gpu.get_lock():
        gpu_index = next_gpu.value % len(GPU_IDS)
        next_gpu.value += 1
    # Pin the selected GPU for this process
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[gpu_index])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"
    logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")


def init_converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions

    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def init_tokenizer():
    try:
        import tiktoken
        tok = tiktoken.get_encoding("gpt2")
        return lambda text: len(tok.encode(text)) if text else 0
    except Exception:
        return lambda text: 0


def extract_pdf_with_docling(pdf_path):
    converter = init_converter()
    count_tokens = init_tokenizer()

    from docling_core.types.doc import TextItem
    from docling_core.types.doc.labels import DocItemLabel

    try:
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        text_md = doc.export_to_markdown()
        for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
            text_md = text_md.replace(artifact, "")

        formula_list = [
            {"latex": el.text}
            for el in doc.texts
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA
        ]

        all_tables_json = [
            table.export_to_dataframe().to_dict(orient="records")
            for table in doc.tables
        ]

        token_count = count_tokens(text_md)

        return {
            "FullText": text_md,
            "TablesJson": json.dumps(all_tables_json, ensure_ascii=False),
            "EquationsJson": json.dumps(formula_list, ensure_ascii=False),
            "TokenCount": token_count,
            "Error": None
        }

    except Exception as e:
        logging.error(f"Docling extraction failed for {pdf_path}: {e}", exc_info=True)
        return {
            "FullText": "ANALYSIS_ERROR",
            "TablesJson": "ANALYSIS_ERROR",
            "EquationsJson": "ANALYSIS_ERROR",
            "TokenCount": 0,
            "Error": str(e)
        }


def do_docling_extraction(df: pd.DataFrame, max_workers: int = 2) -> pd.DataFrame:
    """
    Multiprocessing extraction using a worker initializer to pin each process to a GPU.
    """
    logging.info(f"Starting multiprocessing Docling extraction for {len(df)} records using {max_workers} worker(s)")
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing)...")

    # ensure output columns
    for col in ["FullText", "TablesJson", "EquationsJson", "TokenCount", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else ""

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        for idx, row in df.iterrows():
            pdf_path = row.get("PDFPath", "")
            if pdf_path and os.path.exists(pdf_path):
                futures[executor.submit(extract_pdf_with_docling, pdf_path)] = idx
                print(f" -> Submitted PDF row {idx}, file: {os.path.basename(pdf_path)}")
            else:
                logging.warning(f"Row {idx} skipped: PDF path not found or empty.")
                df.at[idx, "Error"] = "PDF_NOT_FOUND"
                df.at[idx, "FullText"] = "ANALYSIS_ERROR"

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                df.at[idx, "FullText"] = result["FullText"]
                df.at[idx, "TablesJson"] = result["TablesJson"]
                df.at[idx, "EquationsJson"] = result["EquationsJson"]
                df.at[idx, "TokenCount"] = result["TokenCount"]
                df.at[idx, "Error"] = result["Error"]

                if result["Error"]:
                    logging.error(f"[!] Extraction error row {idx}: {result['Error']}")
                    print(f"[!] Extraction error row {idx}: {result['Error']}")
                else:
                    logging.info(f"Extraction successful row {idx}, tokens={result['TokenCount']}")
                    print(f"[✓] Extraction successful row {idx}")

            except Exception as e:
                logging.exception(f"Multiprocessing extraction exception row {idx}: {e}")
                df.at[idx, "FullText"] = "ANALYSIS_ERROR"
                df.at[idx, "TablesJson"] = "ANALYSIS_ERROR"
                df.at[idx, "EquationsJson"] = "ANALYSIS_ERROR"
                df.at[idx, "TokenCount"] = 0
                df.at[idx, "Error"] = str(e)

    logging.info("Multiprocessing Docling extraction complete.")
    return df
