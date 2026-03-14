# docling_extract_formulas_debug_mp_multi_provenance.py

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import logging
import json
import tiktoken
import re
# import any other modules that are GPU related only after the workers are initialized

# List of GPU IDs you want to use:
GPU_IDS = [0, 3]

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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    # Configure logging for worker process
    import logging
    LOG_FILE = "/mnt/c/Users/WSTATION/Desktop/docling_mods/scripts/logs/docling_testing.log"

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    # Add file handler
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
        num_threads=14,
        device=AcceleratorDevice.CUDA
    )

    # Configure layout model
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

    # Enable debug visualization for bounding boxes
    settings.debug.visualize_raw_layout = True
    settings.debug.visualize_layout = True
    settings.debug.debug_output_path = "/mnt/c/Users/WSTATION/Desktop/docling_mods/docling_debug"

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

    # Check for reference section headers
    header_patterns = [
        r'^\s*references?\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*works\s+cited\s*$',
        r'^\s*literature\s+cited\s*$',
    ]

    for pattern in header_patterns:
        if re.search(pattern, text_lower, re.MULTILINE):
            return True

    # Check for citation density (multiple numbered citations)
    citation_pattern = r'^\s*\d+\.\s+[A-Z][^.]+\.\s+\(\d{4}\)'
    citation_matches = re.findall(citation_pattern, text, re.MULTILINE)

    # If more than 3 citations on a page, likely references
    if len(citation_matches) >= 3:
        return True

    return False

def extract_page_metadata(doc, page_no, count_tokens):
    """
    Extract content and metadata for a single page.

    Returns dict with:
        - page_no: int
        - content: markdown string
        - content_before_references: markdown before references section
        - reference_content: markdown of references section (if any)
        - has_references: bool - does this page contain a references section
        - token_count: int
        - items: list of items with provenance (page_no, bbox, label, is_reference)
    """
    from docling_core.types.doc.labels import DocItemLabel

    # Extract items with provenance for this page
    page_items = []
    in_references = False

    for item, level in doc.iterate_items(page_no=page_no, with_groups=False):
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if prov.page_no == page_no:
                    item_label = item.label.value if hasattr(item, 'label') else "unknown"
                    item_text = item.text if hasattr(item, 'text') else ""

                    # Check if this is the start of references section
                    if item_label == "section_header":
                        text_lower = item_text.lower().strip()
                        if any(term in text_lower for term in ['references', 'bibliography', 'works cited', 'literature cited']):
                            in_references = True

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
                        "is_reference": in_references
                    }
                    page_items.append(item_data)
                    break  # Only take first prov for this page

    # Export full page content
    page_md = doc.export_to_markdown(page_no=page_no)

    # Clean artifacts
    artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
    for artifact in artifacts_to_remove:
        page_md = page_md.replace(artifact, "")

    # Separate content before and after references
    content_before_refs = []
    reference_content = []

    for item in page_items:
        if item["is_reference"]:
            reference_content.append(item["text"])
        else:
            content_before_refs.append(item["text"])

    content_before_refs_text = "\n\n".join(content_before_refs)
    reference_content_text = "\n\n".join(reference_content)

    has_references = len(reference_content) > 0

    # Token counts
    token_count = count_tokens(page_md)
    token_count_before_refs = count_tokens(content_before_refs_text) if content_before_refs_text else 0
    token_count_refs = count_tokens(reference_content_text) if reference_content_text else 0

    return {
        "page_no": page_no,
        "content": page_md,  # Full content
        "content_before_references": content_before_refs_text,
        "reference_content": reference_content_text,
        "has_references": has_references,
        "token_count": token_count,
        "token_count_before_references": token_count_before_refs,
        "token_count_references": token_count_refs,
        "items": page_items
    }


def extract_pdf_with_docling(pdf_path: str, idx: int, output_dir=None) -> dict:
    """
    In the child process, reinitialize the debug pipeline, convert the PDF,
    parse formulas, and return the results with page-level provenance.
    """
    count_tokens = init_tokenizer()
    converter = init_debug_pipeline()

    try:
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document

        # Extract document metadata
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)

        # Full markdown export
        text_md = doc.export_to_markdown()

        # Clean artifacts from full text
        artifacts_to_remove = ["<!-- image -->", "$$MALFORMED_FORMULA$$"]
        for artifact in artifacts_to_remove:
            text_md = text_md.replace(artifact, "")

        # Extract page-level content with provenance
        pages_content = []
        for page_no in range(num_pages):
            page_metadata = extract_page_metadata(doc, page_no, count_tokens)
            pages_content.append(page_metadata)

        # Get formulas from the document
        formula_list = []
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel
        for el in doc.texts:
            if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA and el.text != "$$MALFORMED_FORMULA$$":
                # Add provenance to formula
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

        # Process tables with provenance
        import warnings
        all_tables_json = []
        for table in doc.tables:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="DataFrame columns are not unique")
                table_data = {
                    "data": table.export_to_dataframe(doc=doc).to_dict(orient="records")
                }
                # Add table provenance
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
                all_tables_json.append(table_data)

        token_count = count_tokens(text_md)

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
        logging.error(f"Docling extraction failed for {pdf_path}: {e}", exc_info=True)
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
    This version uses the worker_initializer to distribute GPUs.
    """
    logging.info("Starting multiprocessing docling extraction on %d records using max_workers=%d", len(df), max_workers)
    print("[Step 9/11] Extracting text/tables/formulas via Docling (Multiprocessing Dual GPU Debug with Provenance)...")
    for col in ["FullText", "PagesJson", "TablesJson", "EquationsJson", "TokenCount", "NumPages", "NumTables", "NumPictures", "Error"]:
        if col not in df.columns:
            df[col] = None if col == "Error" else (0 if col.startswith("Num") else "")

    futures = {}
    # The initializer ensures each worker gets its GPU assigned
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
                    print(f"[✓] Extraction successful row {irow}: {result['NumPages']} pages, {result['NumTables']} tables")

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


# if __name__ == "__main__":
#     import sys
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )
#
#     TEST_PATH = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/docling_test_30.feather"
#     OUTPUT_PATH = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/enriched_output_debug_mproc_provenance.feather"
#
#     if len(sys.argv) > 1:
#         TEST_PATH = sys.argv[1]
#     if len(sys.argv) > 2:
#         OUTPUT_PATH = sys.argv[2]
#
#     df = pd.read_feather(TEST_PATH)
#     logging.info(f"Loaded {len(df)} rows from {TEST_PATH}")
#
#     # Set max_workers to 2 if you have 2 dedicated GPUs
#     df = do_docling_extraction(df, max_workers=2)
#
#     df.to_feather(OUTPUT_PATH)
#     logging.info(f"Saved results to {OUTPUT_PATH}")
