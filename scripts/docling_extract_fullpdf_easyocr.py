import argparse
import json
import logging
import multiprocessing
import os
import re
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
from pydantic import BaseModel, ConfigDict, ValidationError

# list of gpu ids intended for this workflow
GPU_IDS = [2, 4]

# global variables for assigning gpus
assign_lock = multiprocessing.Lock()
next_gpu = multiprocessing.Value("i", 0)

# per-worker globals
_worker_converter = None
_worker_tokenizer = None

# number of threads for post-processing within a single pdf
POST_PROCESS_WORKERS = 12


OUTPUT_COLUMNS = [
    "PDFPath",
    "FileName",
    "FullText",
    "PagesJson",
    "TablesJson",
    "EquationsJson",
    "TokenCount",
    "NumPages",
    "NumTables",
    "NumPictures",
    "Error",
]
RESULT_COLUMNS = OUTPUT_COLUMNS[2:]


class ExtractionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PDFPath: str
    FileName: str
    FullText: str
    PagesJson: str
    TablesJson: str
    EquationsJson: str
    TokenCount: int
    NumPages: int
    NumTables: int
    NumPictures: int
    Error: str | None


def configure_runtime(gpu_id: int) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"


def init_tokenizer():
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        return lambda text: len(tokenizer.encode(text)) if text else 0
    except Exception:
        return lambda text: 0


def init_converter(debug_output_dir: Path | None, ocr_engine: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        DOCLING_LAYOUT_HERON_101,
        EasyOcrOptions,
        LayoutOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, PdfFormatOption

    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA,
    )

    layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_HERON_101)
    if ocr_engine == "easyocr":
        ocr_options = EasyOcrOptions(
            lang=["en"],
            force_full_page_ocr=True,
            download_enabled=False,
        )
    elif ocr_engine == "rapidocr":
        ocr_options = RapidOcrOptions(
            lang=["english"],
            force_full_page_ocr=True,
            backend="torch",
        )
    else:
        raise ValueError(f"Unsupported ocr engine: {ocr_engine}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.layout_options = layout_options
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.images_scale = 2.0

    if debug_output_dir is not None:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        settings.debug.visualize_raw_layout = True
        settings.debug.visualize_layout = True
        settings.debug.debug_output_path = str(debug_output_dir)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def worker_initializer(
    gpu_ids: list[int],
    debug_output_dir_str: str | None,
    log_file_path: str | None,
    ocr_engine: str,
) -> None:
    global _worker_converter, _worker_tokenizer

    with assign_lock, next_gpu.get_lock():
        gpu_index = next_gpu.value % len(gpu_ids)
        next_gpu.value += 1

    gpu_id = gpu_ids[gpu_index]
    configure_runtime(gpu_id)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
            )
        )
        root_logger.addHandler(file_handler)

    logging.info(
        "worker %s assigned gpu:%s",
        multiprocessing.current_process().name,
        os.environ["CUDA_VISIBLE_DEVICES"],
    )

    debug_output_dir = (
        Path(debug_output_dir_str).expanduser().resolve()
        if debug_output_dir_str
        else None
    )

    try:
        _worker_converter = init_converter(debug_output_dir, ocr_engine)
        _worker_tokenizer = init_tokenizer()
        logging.info(
            "worker %s converter and tokenizer initialized for ocr_engine=%s",
            multiprocessing.current_process().name,
            ocr_engine,
        )
    except Exception:
        logging.exception(
            "worker %s failed to initialize converter/tokenizer",
            multiprocessing.current_process().name,
        )
        raise


def is_reference_section(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False

    text_lower = text.lower()
    header_patterns = [
        r"^\s*references?\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*works\s+cited\s*$",
        r"^\s*literature\s+cited\s*$",
    ]

    for pattern in header_patterns:
        if re.search(pattern, text_lower, re.MULTILINE):
            return True

    citation_matches = re.findall(
        r"^\s*\d+\.\s+[A-Z][^.]+\.\s+\(\d{4}\)",
        text,
        re.MULTILINE,
    )
    return len(citation_matches) >= 3


def build_all_page_items(doc) -> dict[int, list[dict[str, Any]]]:
    page_items_map: dict[int, list[dict[str, Any]]] = {}

    for item, _ in doc.iterate_items(with_groups=False):
        if not hasattr(item, "prov") or not item.prov:
            continue
        for prov in item.prov:
            item_label = item.label.value if hasattr(item, "label") else "unknown"
            item_text = item.text if hasattr(item, "text") else ""

            item_data = {
                "label": item_label,
                "text": item_text,
                "page_no": prov.page_no,
                "bbox": {
                    "l": prov.bbox.l,
                    "t": prov.bbox.t,
                    "r": prov.bbox.r,
                    "b": prov.bbox.b,
                }
                if prov.bbox
                else None,
                "charspan": list(prov.charspan) if prov.charspan else None,
                "is_reference": False,
            }

            page_items_map.setdefault(prov.page_no, []).append(item_data)
            break

    return page_items_map


def render_table_markdown(table_data: list[dict[str, Any]]) -> str:
    if not table_data:
        return ""

    cols = list(table_data[0].keys())
    lines = []
    lines.append("| " + " | ".join(str(col) for col in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in table_data:
        cells = [
            str(row.get(col, "")).replace("|", "/").replace("\n", " ")
            for col in cols
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def match_table_to_item(
    item_bbox: dict[str, float] | None,
    page_tables: list[dict],
) -> dict | None:
    if not page_tables or not item_bbox:
        return None

    item_cx = (item_bbox["l"] + item_bbox["r"]) / 2
    item_cy = (item_bbox["t"] + item_bbox["b"]) / 2
    best_table = None
    best_dist = float("inf")

    for table_data in page_tables:
        for prov in table_data.get("provenance", []):
            bbox = prov.get("bbox")
            if not bbox:
                continue
            table_cx = (bbox["l"] + bbox["r"]) / 2
            table_cy = (bbox["t"] + bbox["b"]) / 2
            dist = ((item_cx - table_cx) ** 2 + (item_cy - table_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_table = table_data

    return best_table


def build_page_metadata(
    page_no: int,
    page_items: list[dict],
    count_tokens,
    page_tables: list[dict],
) -> dict[str, Any]:
    for item in page_items:
        if item["label"] == "table":
            matched_table = match_table_to_item(item.get("bbox"), page_tables)
            if matched_table:
                item["text"] = matched_table.get("data", [])

    in_references = False
    for item in page_items:
        if item["label"] == "section_header" and is_reference_section(item["text"]):
            in_references = True
        item["is_reference"] = in_references

    all_texts = []
    for item in page_items:
        if isinstance(item["text"], list):
            text = render_table_markdown(item["text"])
        elif isinstance(item["text"], str):
            text = item["text"]
        else:
            text = ""
        if text:
            all_texts.append(text)
    page_md = "\n\n".join(all_texts)

    for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
        page_md = page_md.replace(artifact, "")

    content_before_refs = []
    reference_content = []
    for item in page_items:
        if isinstance(item["text"], list):
            text = render_table_markdown(item["text"])
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

    before_refs = "\n\n".join(content_before_refs)
    ref_text = "\n\n".join(reference_content)

    return {
        "page_no": page_no,
        "content": page_md,
        "content_before_references": before_refs,
        "reference_content": ref_text,
        "has_references": len(reference_content) > 0,
        "token_count": count_tokens(page_md),
        "token_count_before_references": count_tokens(before_refs),
        "token_count_references": count_tokens(ref_text),
        "items": page_items,
    }


def export_single_table(table, doc) -> dict[str, Any]:
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
                        "b": prov.bbox.b,
                    }
                    if prov.bbox
                    else None,
                }
                for prov in table.prov
            ]
        return table_data


def default_result(error: str | None = None) -> dict[str, Any]:
    return {
        "FullText": "ANALYSIS_ERROR" if error else "",
        "PagesJson": "ANALYSIS_ERROR" if error else "",
        "TablesJson": "ANALYSIS_ERROR" if error else "",
        "EquationsJson": "ANALYSIS_ERROR" if error else "",
        "TokenCount": 0,
        "NumPages": 0,
        "NumTables": 0,
        "NumPictures": 0,
        "Error": error,
    }


def initialize_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = default_result()[col]
    return out


def validate_output_row(
    pdf_path: str,
    file_name: str,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    try:
        row = ExtractionRow(PDFPath=pdf_path, FileName=file_name, **raw_result)
    except ValidationError as exc:
        row = ExtractionRow(
            PDFPath=pdf_path,
            FileName=file_name,
            **default_result(f"VALIDATION_ERROR: {exc}"),
        )
    return row.model_dump()


def assign_row_result(df: pd.DataFrame, row_index: Any, result: dict[str, Any]) -> None:
    for col in OUTPUT_COLUMNS:
        df.at[row_index, col] = result[col]


def row_is_complete(row: pd.Series) -> bool:
    return row.get("Error") is None and bool(row.get("FullText"))


def row_is_timeout(row: pd.Series) -> bool:
    return row.get("Error") == "TIMEOUT"


def row_should_retry(row: pd.Series, retry_errors: bool) -> bool:
    if row_is_complete(row):
        return False
    if row_is_timeout(row):
        return True
    error = row.get("Error")
    if error == "IN_PROGRESS":
        return True
    if error in (None, "", 0):
        return True
    return retry_errors


def load_or_initialize_dataframe(
    input_df: pd.DataFrame,
    output_path: Path | None,
    resume: bool,
    retry_errors: bool,
) -> pd.DataFrame:
    base = initialize_output_dataframe(input_df)
    if not resume or output_path is None or not output_path.exists():
        return base

    existing = pd.read_feather(output_path)
    missing = [col for col in OUTPUT_COLUMNS if col not in existing.columns]
    if missing:
        raise ValueError(f"Existing resume file is missing columns: {missing}")
    if len(existing) != len(base):
        raise ValueError("Existing resume file row count does not match input row count")
    if (
        existing["PDFPath"].tolist() != base["PDFPath"].tolist()
        or existing["FileName"].tolist() != base["FileName"].tolist()
    ):
        raise ValueError("Existing resume file does not match input PDF ordering")

    for idx in base.index:
        row = existing.loc[idx]
        if row_is_complete(row) or row_should_retry(row, retry_errors):
            for col in RESULT_COLUMNS:
                base.at[idx, col] = row[col]
        elif row.get("Error"):
            for col in RESULT_COLUMNS:
                base.at[idx, col] = row[col]

    return base


def write_checkpoint(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    df.to_feather(tmp_path)
    os.replace(tmp_path, output_path)


def should_flush_checkpoint(
    dirty_count: int,
    last_save_time: float,
    now: float,
    save_every: int,
    save_interval_seconds: float,
) -> bool:
    if dirty_count <= 0:
        return False
    if dirty_count >= max(1, save_every):
        return True
    return (now - last_save_time) >= save_interval_seconds


def create_executor(
    max_workers: int,
    gpu_ids: list[int],
    debug_output_dir: Path | None,
    log_file_path: str | None,
    ocr_engine: str,
):
    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_initializer,
        initargs=(
            gpu_ids,
            str(debug_output_dir) if debug_output_dir else None,
            log_file_path,
            ocr_engine,
        ),
    )


def shutdown_executor(executor, *, kill_workers: bool) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=not kill_workers, cancel_futures=True)
    finally:
        if kill_workers:
            processes = getattr(executor, "_processes", None) or {}
            for proc in processes.values():
                if proc.is_alive():
                    proc.terminate()
            for proc in processes.values():
                proc.join(timeout=5)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()


def pending_row_indices(df: pd.DataFrame, retry_errors: bool) -> list[Any]:
    return [idx for idx in df.index if row_should_retry(df.loc[idx], retry_errors)]


def mark_missing_pdf(
    df: pd.DataFrame,
    row_index: Any,
    pdf_path: str,
    file_name: str,
) -> None:
    result = validate_output_row(pdf_path, file_name, default_result("PDF_NOT_FOUND"))
    assign_row_result(df, row_index, result)


def mark_timeout(
    df: pd.DataFrame,
    row_index: Any,
    pdf_path: str,
    file_name: str,
) -> None:
    result = validate_output_row(pdf_path, file_name, default_result("TIMEOUT"))
    assign_row_result(df, row_index, result)


def mark_in_progress(
    df: pd.DataFrame,
    row_index: Any,
    pdf_path: str,
    file_name: str,
) -> None:
    result = validate_output_row(
        pdf_path,
        file_name,
        {
            "FullText": "",
            "PagesJson": "",
            "TablesJson": "",
            "EquationsJson": "",
            "TokenCount": 0,
            "NumPages": 0,
            "NumTables": 0,
            "NumPictures": 0,
            "Error": "IN_PROGRESS",
        },
    )
    assign_row_result(df, row_index, result)


def collect_pdf_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    if input_path.is_dir():
        pdf_paths = sorted(input_path.rglob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDF files found under {input_path}")
        return pdf_paths

    raise ValueError(f"Input path does not exist: {input_path}")


def create_pdf_dataframe(input_path: Path) -> pd.DataFrame:
    pdf_paths = collect_pdf_paths(input_path)
    return pd.DataFrame(
        {
            "PDFPath": [str(path) for path in pdf_paths],
            "FileName": [path.name for path in pdf_paths],
        }
    )


def parse_gpu_ids(gpu_ids_arg: str) -> list[int]:
    gpu_ids = []
    for chunk in gpu_ids_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gpu_ids.append(int(chunk))

    if not gpu_ids:
        raise ValueError("--gpu-ids did not contain any gpu ids.")

    return gpu_ids


def extract_pdf_with_docling(
    pdf_path: str,
    idx: int,
    debug_output_dir: str | None = None,
) -> dict[str, Any]:
    global _worker_converter, _worker_tokenizer

    t_total_start = time.time()
    logging.info("[Row %s] Starting OCR extraction for %s", idx, pdf_path)

    converter = _worker_converter
    count_tokens = _worker_tokenizer

    if converter is None or count_tokens is None:
        error = "WORKER_NOT_INITIALIZED"
        logging.error("[Row %s] %s", idx, error)
        return default_result(error)

    try:
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel

        t0 = time.time()
        conv_res = converter.convert(pdf_path)
        conv_res.document.name = f"pdf_row_{idx}"
        doc = conv_res.document
        num_pages = len(doc.pages)
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)
        logging.info(
            "[Row %s] Step 1/6: OCR Docling conversion done in %.2fs — %s pages, %s tables, %s pictures",
            idx,
            time.time() - t0,
            num_pages,
            num_tables,
            num_pictures,
        )

        t0 = time.time()
        text_md = doc.export_to_markdown()
        for artifact in ["<!-- image -->", "$$MALFORMED_FORMULA$$"]:
            text_md = text_md.replace(artifact, "")
        logging.info(
            "[Row %s] Step 2/6: Full markdown export done in %.2fs (%s chars)",
            idx,
            time.time() - t0,
            len(text_md),
        )

        t0 = time.time()
        all_tables_json = []
        if num_tables > 0:
            table_export_fn = partial(export_single_table, doc=doc)
            with ThreadPoolExecutor(
                max_workers=min(POST_PROCESS_WORKERS, num_tables)
            ) as executor:
                futures = {
                    executor.submit(table_export_fn, table): table_idx
                    for table_idx, table in enumerate(doc.tables)
                }
                table_results = [None] * num_tables
                for future in as_completed(futures):
                    table_idx = futures[future]
                    try:
                        table_results[table_idx] = future.result()
                    except Exception as exc:
                        logging.error(
                            "[Row %s] Table %s export failed: %s",
                            idx,
                            table_idx,
                            exc,
                            exc_info=True,
                        )
                        table_results[table_idx] = {"data": [], "error": str(exc)}
                all_tables_json = table_results
        logging.info(
            "[Row %s] Step 3/6: Table export done in %.2fs (%s tables)",
            idx,
            time.time() - t0,
            num_tables,
        )

        tables_by_page: dict[int, list[dict[str, Any]]] = {}
        for table_data in all_tables_json:
            if not table_data:
                continue
            for prov in table_data.get("provenance", []):
                page_no = prov.get("page_no")
                if page_no is not None:
                    tables_by_page.setdefault(page_no, []).append(table_data)

        t0 = time.time()
        page_items_map = build_all_page_items(doc)
        total_items = sum(len(v) for v in page_items_map.values())
        logging.info(
            "[Row %s] Step 4a/6: Single-pass item bucketing done in %.2fs (%s items across %s pages)",
            idx,
            time.time() - t0,
            total_items,
            len(page_items_map),
        )

        t0 = time.time()
        pages_content = [None] * num_pages
        with ThreadPoolExecutor(max_workers=POST_PROCESS_WORKERS) as executor:
            futures = {
                executor.submit(
                    build_page_metadata,
                    page_no,
                    page_items_map.get(page_no, []),
                    count_tokens,
                    tables_by_page.get(page_no, []),
                ): page_no
                for page_no in range(num_pages)
            }
            done_count = 0
            for future in as_completed(futures):
                page_no = futures[future]
                try:
                    pages_content[page_no] = future.result()
                except Exception as exc:
                    logging.error(
                        "[Row %s] Page %s metadata build failed: %s",
                        idx,
                        page_no,
                        exc,
                        exc_info=True,
                    )
                    pages_content[page_no] = {
                        "page_no": page_no,
                        "content": "",
                        "content_before_references": "",
                        "reference_content": "",
                        "has_references": False,
                        "token_count": 0,
                        "token_count_before_references": 0,
                        "token_count_references": 0,
                        "items": [],
                        "error": str(exc),
                    }
                done_count += 1
                if done_count % 200 == 0 or done_count == num_pages:
                    logging.info(
                        "[Row %s] Step 4b/6: Page metadata progress %s/%s",
                        idx,
                        done_count,
                        num_pages,
                    )
        logging.info(
            "[Row %s] Step 4b/6: Page metadata build done in %.2fs (%s pages, %s threads)",
            idx,
            time.time() - t0,
            num_pages,
            POST_PROCESS_WORKERS,
        )

        in_refs = False
        for page in pages_content:
            if page is None:
                continue
            for item in page.get("items", []):
                if item["label"] == "section_header":
                    text_lower = item["text"].lower() if isinstance(item["text"], str) else ""
                    if any(
                        term in text_lower
                        for term in [
                            "references",
                            "bibliography",
                            "works cited",
                            "literature cited",
                        ]
                    ):
                        in_refs = True
                item["is_reference"] = in_refs
            if in_refs:
                ref_texts = []
                non_ref_texts = []
                for item in page.get("items", []):
                    if isinstance(item["text"], list):
                        text = render_table_markdown(item["text"])
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
                page["token_count_before_references"] = count_tokens(
                    page["content_before_references"]
                )
                page["token_count_references"] = count_tokens(
                    page["reference_content"]
                )

        t0 = time.time()
        formula_list = []
        for element in doc.texts:
            if (
                isinstance(element, TextItem)
                and element.label == DocItemLabel.FORMULA
                and element.text != "$$MALFORMED_FORMULA$$"
            ):
                formula_data = {"latex": element.text}
                if element.prov:
                    formula_data["provenance"] = [
                        {
                            "page_no": prov.page_no,
                            "bbox": {
                                "l": prov.bbox.l,
                                "t": prov.bbox.t,
                                "r": prov.bbox.r,
                                "b": prov.bbox.b,
                            }
                            if prov.bbox
                            else None,
                        }
                        for prov in element.prov
                    ]
                formula_list.append(formula_data)
        logging.info(
            "[Row %s] Step 5/6: Formula extraction done in %.2fs (%s formulas)",
            idx,
            time.time() - t0,
            len(formula_list),
        )

        t0 = time.time()
        token_count = count_tokens(text_md)
        result = {
            "FullText": text_md,
            "PagesJson": json.dumps(pages_content, ensure_ascii=False),
            "TablesJson": json.dumps(all_tables_json, ensure_ascii=False),
            "EquationsJson": json.dumps(formula_list, ensure_ascii=False),
            "TokenCount": token_count,
            "NumPages": num_pages,
            "NumTables": num_tables,
            "NumPictures": num_pictures,
            "Error": None,
        }
        logging.info(
            "[Row %s] Step 6/6: Final serialization done in %.2fs (%s tokens)",
            idx,
            time.time() - t0,
            token_count,
        )
        logging.info(
            "[Row %s] OCR extraction complete — total %.2fs",
            idx,
            time.time() - t_total_start,
        )
        return result

    except Exception as exc:
        logging.error(
            "[Row %s] OCR Docling extraction failed for %s: %s",
            idx,
            pdf_path,
            exc,
            exc_info=True,
        )
        return default_result(str(exc))


def do_docling_extraction(
    df: pd.DataFrame,
    *,
    gpu_ids: list[int],
    ocr_engine: str,
    max_workers: int,
    debug_output_dir: Path | None,
    log_file_path: str | None,
    output_path: str | None,
    save_every: int,
    save_interval_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
    resume: bool,
    retry_errors: bool,
    clock_fn=time.monotonic,
) -> pd.DataFrame:
    logging.info(
        "Starting multiprocessing OCR extraction on %d records using max_workers=%d",
        len(df),
        max_workers,
    )

    checkpoint_path = Path(output_path) if output_path else None
    df = load_or_initialize_dataframe(
        df,
        checkpoint_path,
        resume=resume,
        retry_errors=retry_errors,
    )
    dirty_count = 0
    last_save_time = clock_fn()

    rows_to_process = pending_row_indices(df, retry_errors=retry_errors)
    pending: list[Any] = []
    for idx in rows_to_process:
        pdf_path = df.at[idx, "PDFPath"]
        if not pdf_path or not os.path.exists(pdf_path):
            logging.warning("[MainProc] Missing PDF path for row %s: %s", idx, pdf_path)
            mark_missing_pdf(df, idx, str(pdf_path), str(df.at[idx, "FileName"]))
            dirty_count += 1
            now = clock_fn()
            if checkpoint_path and should_flush_checkpoint(
                dirty_count,
                last_save_time,
                now,
                save_every,
                save_interval_seconds,
            ):
                write_checkpoint(df, checkpoint_path)
                dirty_count = 0
                last_save_time = now
        else:
            pending.append(idx)

    executor = None
    in_flight: dict[Any, tuple[int, float]] = {}

    try:
        executor = create_executor(
            max_workers=max_workers,
            gpu_ids=gpu_ids,
            debug_output_dir=debug_output_dir,
            log_file_path=log_file_path,
            ocr_engine=ocr_engine,
        )

        while pending or in_flight:
            while executor is not None and pending and len(in_flight) < max_workers:
                idx = pending.pop(0)
                pdf_path = df.at[idx, "PDFPath"]
                mark_in_progress(df, idx, str(pdf_path), str(df.at[idx, "FileName"]))
                dirty_count += 1
                logging.info("[MainProc] Submitting row %s, pdf=%s", idx, pdf_path)
                future = executor.submit(extract_pdf_with_docling, pdf_path, idx)
                in_flight[future] = (idx, clock_fn())

                now = clock_fn()
                if checkpoint_path and should_flush_checkpoint(
                    dirty_count,
                    last_save_time,
                    now,
                    save_every,
                    save_interval_seconds,
                ):
                    write_checkpoint(df, checkpoint_path)
                    dirty_count = 0
                    last_save_time = now

            if not in_flight:
                break

            done, _ = wait(
                set(in_flight.keys()),
                timeout=poll_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            now = clock_fn()

            if not done:
                overdue = [
                    future
                    for future, (_, started_at) in in_flight.items()
                    if now - started_at >= task_timeout_seconds
                ]
                if overdue:
                    timed_out_rows = {in_flight[future][0] for future in overdue}
                    lost_rows = [
                        idx
                        for future, (idx, _) in in_flight.items()
                        if future not in overdue
                    ]

                    for idx in sorted(timed_out_rows):
                        logging.error(
                            "[MainProc] Timeout row %s after %.1fs",
                            idx,
                            task_timeout_seconds,
                        )
                        mark_timeout(
                            df,
                            idx,
                            str(df.at[idx, "PDFPath"]),
                            str(df.at[idx, "FileName"]),
                        )
                        dirty_count += 1

                    if checkpoint_path:
                        write_checkpoint(df, checkpoint_path)
                        dirty_count = 0
                        last_save_time = now

                    shutdown_executor(executor, kill_workers=True)
                    executor = create_executor(
                        max_workers=max_workers,
                        gpu_ids=gpu_ids,
                        debug_output_dir=debug_output_dir,
                        log_file_path=log_file_path,
                        ocr_engine=ocr_engine,
                    )
                    in_flight.clear()
                    pending = lost_rows + pending
                    continue

                if checkpoint_path and should_flush_checkpoint(
                    dirty_count,
                    last_save_time,
                    now,
                    save_every,
                    save_interval_seconds,
                ):
                    write_checkpoint(df, checkpoint_path)
                    dirty_count = 0
                    last_save_time = now
                continue

            for future in done:
                idx, _ = in_flight.pop(future)
                pdf_path = str(df.at[idx, "PDFPath"])
                file_name = str(df.at[idx, "FileName"])
                try:
                    raw_result = future.result()
                except Exception as exc:
                    logging.exception(
                        "Multiprocessing OCR extraction exception row %s: %s",
                        idx,
                        exc,
                    )
                    raw_result = default_result(str(exc))

                result = validate_output_row(pdf_path, file_name, raw_result)
                assign_row_result(df, idx, result)
                dirty_count += 1

                if result["Error"]:
                    logging.error("[!] Extraction error row %s: %s", idx, result["Error"])
                    print(f"[!] Extraction error row {idx}: {result['Error']}")
                else:
                    logging.info(
                        "Extraction successful row %s, pages=%s, tables=%s, tokens=%s",
                        idx,
                        result["NumPages"],
                        result["NumTables"],
                        result["TokenCount"],
                    )
                    print(
                        f"[OK] Extraction successful row {idx}: "
                        f"{result['NumPages']} pages, {result['NumTables']} tables"
                    )

            now = clock_fn()
            if checkpoint_path and should_flush_checkpoint(
                dirty_count,
                last_save_time,
                now,
                save_every,
                save_interval_seconds,
            ):
                write_checkpoint(df, checkpoint_path)
                dirty_count = 0
                last_save_time = now

    finally:
        shutdown_executor(executor, kill_workers=False)

    if checkpoint_path:
        write_checkpoint(df, checkpoint_path)
    logging.info("Multiprocessing OCR extraction complete.")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract full-PDF Docling outputs using full-page EasyOCR with "
            "checkpointing and resume support."
        )
    )
    parser.add_argument("input_path", help="PDF file or directory of PDFs.")
    parser.add_argument("output_feather", help="Output feather path.")
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "rapidocr"],
        default="easyocr",
        help="OCR backend to use for full-page OCR.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=",".join(str(gpu_id) for gpu_id in GPU_IDS),
        help=(
            "Comma-separated gpu ids reserved for this workflow. "
            f"Default: {GPU_IDS}"
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "Specific gpu id to use exclusively. "
            "If provided, overrides --gpu-ids with a single gpu."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes. Default: 5 per selected gpu.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=1200.0,
        help="Per-task timeout before executor restart.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Supervisor poll interval.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Checkpoint after this many row updates.",
    )
    parser.add_argument(
        "--save-interval-seconds",
        type=float,
        default=30.0,
        help="Checkpoint if this much time passes with unsaved progress.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output feather if present.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, retry rows with non-timeout errors.",
    )
    parser.add_argument(
        "--debug-output-dir",
        default=None,
        help="Optional debug image output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_feather = Path(args.output_feather).expanduser().resolve()
    debug_output_dir = (
        Path(args.debug_output_dir).expanduser().resolve()
        if args.debug_output_dir
        else None
    )

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"fullpdf_{args.ocr_engine}_{timestamp}.log"

    logging.basicConfig(
        filename=str(log_file),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(console_handler)

    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except Exception as exc:
        logging.error(str(exc))
        return 1

    if args.gpu_id is not None:
        gpu_ids = [args.gpu_id]

    max_workers = args.max_workers if args.max_workers is not None else len(gpu_ids) * 5

    logging.info("Input path: %s", input_path)
    logging.info("Output feather: %s", output_feather)
    logging.info("Log file: %s", log_file)
    logging.info("Configured gpu ids: %s", gpu_ids)
    logging.info("OCR engine: %s", args.ocr_engine)
    logging.info("Using max_workers: %s", max_workers)
    logging.info("Resume: %s", args.resume)
    logging.info("Retry errors: %s", args.retry_errors)

    try:
        df = create_pdf_dataframe(input_path)
    except Exception as exc:
        logging.error(str(exc))
        return 1

    logging.info("Found %s pdfs in %s", len(df), input_path)
    print(f"Found {len(df)} PDFs in {input_path}")
    print(df[["PDFPath", "FileName"]].to_string())

    try:
        df = do_docling_extraction(
            df,
            gpu_ids=gpu_ids,
            ocr_engine=args.ocr_engine,
            max_workers=max_workers,
            debug_output_dir=debug_output_dir,
            log_file_path=str(log_file),
            output_path=str(output_feather),
            save_every=args.save_every,
            save_interval_seconds=args.save_interval_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            resume=args.resume,
            retry_errors=args.retry_errors,
        )
    except Exception as exc:
        logging.exception("Full-PDF EasyOCR extraction failed: %s", exc)
        return 1

    logging.info("Saved results to %s", output_feather)
    print(f"Saved results to {output_feather}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
