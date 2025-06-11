# docling_extract_formulas_debug.py
# Multiprocessing Debug Version with Worker Initializer and GPU Pinning
import logging
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Ensure spawn start method to avoid inheriting CUDA context
multiprocessing.set_start_method("spawn", force=True)

# List of GPU IDs you want to use for debug (single-GPU)
GPU_IDS = ["0"]
# Globals for round-robin GPU assignment
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value('i', 0)


def worker_initializer():
    """
    Assigns one GPU from GPU_IDS to this worker process in a round-robin fashion.
    """
    with assign_lock, next_gpu.get_lock():
        gpu_index = next_gpu.value % len(GPU_IDS)
        next_gpu.value += 1
    os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]    = str(GPU_IDS[gpu_index])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"
    logging.info(f"Worker {multiprocessing.current_process().name} assigned GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")


def init_converter(output_dir=None):
    """
    Create a DocumentConverter with debug visualization and formula enrichment.
    """
    from docling.utils.export import generate_multimodal_pages
    from docling_core.types.doc import TextItem
    from docling_core.types.doc.labels import DocItemLabel
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions
    from docling.datamodel.settings import settings

    # Configure debug visuals
    settings.debug.visualize_raw_layout = True
    settings.debug.visualize_layout     = True
    if output_dir:
        settings.debug.debug_output_path = output_dir

    # Set up GPU accelerator
    accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CUDA
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr               = False
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_page_images  = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.images_scale          = 2.0

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

########################################
# 3) Core Extraction Function
########################################

def extract_pdf_with_docling(args):
    """
    Child process: convert a single PDF, extract markdown, tables, formulas, and debug pages.
    args: (pdf_path, row_idx, output_dir)
    Returns: (row_idx, result_dict)
    """
    pdf_path, row_idx, output_dir = args
    converter    = init_converter(output_dir)
    count_tokens = init_tokenizer()

    # imports deferred to avoid early CUDA init
    from docling_core.types.doc import TextItem
    from docling_core.types.doc.labels import DocItemLabel
    from docling.utils.export import generate_multimodal_pages

    try:
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        # Markdown extraction
        text_md = doc.export_to_markdown()

        # Extract formulas
        formula_list = [
            {"latex": el.text}
            for el in doc.texts
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA
        ]

        # Extract tables
        all_tables_json = [
            table.export_to_dataframe().to_dict(orient='records')
            for table in doc.tables
        ]

        # Trigger debug page exports
        for (content_text, content_md, content_dt, page_cells, page_segments, page) in generate_multimodal_pages(conv_res):
            # visuals saved via settings.debug.debug_output_path
            continue

        token_count = count_tokens(text_md)

        return row_idx, {
            "FullText": text_md,
            "TablesJson": json.dumps(all_tables_json, ensure_ascii=False),
            "EquationsJson": json.dumps(formula_list, ensure_ascii=False),
            "TokenCount": token_count,
            "Error": None
        }

    except Exception as e:
        logging.error(f"Debug extraction failed for {pdf_path}: {e}", exc_info=True)
        return row_idx, {
            "FullText": "ANALYSIS_ERROR",
            "TablesJson": "ANALYSIS_ERROR",
            "EquationsJson": "ANALYSIS_ERROR",
            "TokenCount": 0,
            "Error": str(e)
        }

########################################
# 4) Main Loop Over DataFrame
########################################

def do_docling_extraction(df, output_dir=None, max_workers=2):
    """
    Multiprocessing debug extraction: spawns workers with GPU pinning and extracts rows in parallel.
    """
    logging.info(f"Starting debug multiprocessing for {len(df)} records using {max_workers} worker(s)")

    # Ensure columns exist
    for col in ("FullText","TablesJson","EquationsJson","TokenCount","Error"):
        if col not in df.columns:
            df[col] = None if col == "Error" else ""

    # Prepare args list
    args_list = []
    for idx, row in df.iterrows():
        pdf_path = row.get("PDFPath", "")
        if pdf_path and os.path.exists(pdf_path):
            args_list.append((pdf_path, idx, output_dir))
        else:
            df.at[idx, "Error"] = "PDF_NOT_FOUND"

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        futures = {executor.submit(extract_pdf_with_docling, args): args[1] for args in args_list}
        for future in as_completed(futures):
            idx, result = future.result()
            df.at[idx, "FullText"]     = result["FullText"]
            df.at[idx, "TablesJson"]   = result["TablesJson"]
            df.at[idx, "EquationsJson"] = result["EquationsJson"]
            df.at[idx, "TokenCount"]   = result["TokenCount"]
            df.at[idx, "Error"]        = result["Error"]

    logging.info("Debug Docling extraction complete.")
    return df

# ----------------------------------------
# If run directly for testing:
# ----------------------------------------
# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_feather("data/pre_docling.feather")\#
#     enriched_df = do_docling_extraction(df, output_dir="/path/to/debug/output", max_workers=1)
#     enriched_df.to_feather("data/debug_output.feather")
#     print("Debug extraction finished.")
