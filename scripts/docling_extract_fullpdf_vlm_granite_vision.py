import argparse
import base64
import json
import logging
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
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import tiktoken
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# per-worker globals
_worker_tokenizer = None
_worker_api_url = None
_worker_model_name = None
_worker_api_timeout = None
_worker_api_concurrency = None
_worker_debug_output_dir = None

# number of threads for post-processing within a single pdf
POST_PROCESS_WORKERS = 12
VLM_RESPONSE_FORMAT = "doctags"
VLM_PROMPT = "Convert this page to docling."
VLM_MAX_TOKENS = 4096


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


def init_tokenizer():
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        return lambda text: len(tokenizer.encode(text)) if text else 0
    except Exception:
        return lambda text: 0


def clean_model_response(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    for token in [
        "<|end_of_text|>",
        "<|end|>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<pad>",
        "</s>",
        "<s>",
    ]:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def render_pdf_pages(pdf_path: str, scale: float = 2.0) -> list[Image.Image]:
    import pypdfium2 as pdfium

    images: list[Image.Image] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            bitmap = None
            try:
                bitmap = page.render(scale=scale)
                images.append(bitmap.to_pil().convert("RGB"))
            finally:
                if bitmap is not None and hasattr(bitmap, "close"):
                    bitmap.close()
                if hasattr(page, "close"):
                    page.close()
    finally:
        if hasattr(doc, "close"):
            doc.close()

    return images


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_page_doctags(
    *,
    image: Image.Image,
    api_url: str,
    model_name: str,
    api_timeout: float,
) -> str:
    image_b64 = image_to_base64_png(image)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": VLM_PROMPT,
                    },
                ],
            }
        ],
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": 0.0,
        "skip_special_tokens": False,
    }
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=api_timeout,
    )
    response.raise_for_status()
    raw_text = response.json()["choices"][0]["message"]["content"]
    return clean_model_response(raw_text)


def convert_pdf_via_api(
    *,
    pdf_path: str,
    api_url: str,
    model_name: str,
    api_timeout: float,
    api_concurrency: int,
    debug_output_dir: Path | None,
    row_idx: int,
) -> tuple[DoclingDocument, int]:
    images = render_pdf_pages(pdf_path, scale=2.0)
    num_pages = len(images)

    logging.info(
        "[Row %s] Rendered %s page images for direct api conversion",
        row_idx,
        num_pages,
    )

    doctags_by_page: list[str] = [""] * num_pages

    def _call_page(page_number: int, image: Image.Image) -> tuple[int, str]:
        doctags = request_page_doctags(
            image=image,
            api_url=api_url,
            model_name=model_name,
            api_timeout=api_timeout,
        )
        return page_number, doctags

    max_api_workers = max(1, min(api_concurrency, num_pages))
    with ThreadPoolExecutor(max_workers=max_api_workers) as executor:
        futures = {
            executor.submit(_call_page, page_number, image): page_number
            for page_number, image in enumerate(images, start=1)
        }
        for future in as_completed(futures):
            page_number, doctags = future.result()
            doctags_by_page[page_number - 1] = doctags
            logging.info(
                "[Row %s] Direct api page %s/%s returned %s chars",
                row_idx,
                page_number,
                num_pages,
                len(doctags),
            )

    if debug_output_dir is not None:
        debug_pdf_dir = debug_output_dir / f"row_{row_idx}_{Path(pdf_path).stem}"
        debug_pdf_dir.mkdir(parents=True, exist_ok=True)
        for page_number, (image, doctags) in enumerate(
            zip(images, doctags_by_page),
            start=1,
        ):
            image.save(debug_pdf_dir / f"page_{page_number:04d}.png")
            (debug_pdf_dir / f"page_{page_number:04d}.txt").write_text(
                doctags,
                encoding="utf-8",
            )

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        doctags_by_page,
        images,
    )
    doc = DoclingDocument.load_from_doctags(
        doctag_document=doctags_doc,
        document_name=Path(pdf_path).stem,
    )
    return doc, num_pages


def worker_initializer(
    api_url: str,
    model_name: str,
    api_timeout: float,
    api_concurrency: int,
    debug_output_dir_str: str | None,
    log_file_path: str | None,
) -> None:
    global _worker_tokenizer, _worker_api_url, _worker_model_name
    global _worker_api_timeout, _worker_api_concurrency, _worker_debug_output_dir

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

    debug_output_dir = (
        Path(debug_output_dir_str).expanduser().resolve()
        if debug_output_dir_str
        else None
    )

    try:
        _worker_tokenizer = init_tokenizer()
        _worker_api_url = api_url
        _worker_model_name = model_name
        _worker_api_timeout = api_timeout
        _worker_api_concurrency = api_concurrency
        _worker_debug_output_dir = debug_output_dir
        logging.info(
            "worker direct api and tokenizer initialized for response_format=%s endpoint=%s model=%s",
            VLM_RESPONSE_FORMAT,
            api_url,
            model_name,
        )
    except Exception:
        logging.exception("worker failed to initialize direct api/tokenizer")
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


def build_page_metadata(
    page_no: int,
    page_items: list[dict],
    count_tokens,
    page_tables: list[dict],
) -> dict[str, Any]:
    table_index = 0
    for item in page_items:
        if item["label"] == "table" and table_index < len(page_tables):
            item["text"] = page_tables[table_index].get("data", [])
            table_index += 1

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
    *,
    max_workers: int,
    api_url: str,
    model_name: str,
    api_timeout: float,
    api_concurrency: int,
    debug_output_dir: Path | None,
    log_file_path: str | None,
):
    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_initializer,
        initargs=(
            api_url,
            model_name,
            api_timeout,
            api_concurrency,
            str(debug_output_dir) if debug_output_dir else None,
            log_file_path,
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
def extract_pdf_with_docling(
    pdf_path: str,
    idx: int,
) -> dict[str, Any]:
    global _worker_tokenizer, _worker_api_url, _worker_model_name
    global _worker_api_timeout, _worker_api_concurrency, _worker_debug_output_dir

    t_total_start = time.time()
    logging.info("[Row %s] Starting VLM extraction for %s", idx, pdf_path)

    count_tokens = _worker_tokenizer

    if (
        count_tokens is None
        or _worker_api_url is None
        or _worker_model_name is None
        or _worker_api_timeout is None
        or _worker_api_concurrency is None
    ):
        error = "WORKER_NOT_INITIALIZED"
        logging.error("[Row %s] %s", idx, error)
        return default_result(error)

    try:
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel

        t0 = time.time()
        doc, num_pages = convert_pdf_via_api(
            pdf_path=pdf_path,
            api_url=_worker_api_url,
            model_name=_worker_model_name,
            api_timeout=_worker_api_timeout,
            api_concurrency=_worker_api_concurrency,
            debug_output_dir=_worker_debug_output_dir,
            row_idx=idx,
        )
        doc.name = f"pdf_row_{idx}"
        num_tables = len(doc.tables)
        num_pictures = len(doc.pictures)
        logging.info(
            "[Row %s] Step 1/6: Direct api doctags conversion done in %.2fs — %s pages, %s tables, %s pictures",
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
                for page_no in range(1, num_pages + 1)
            }
            done_count = 0
            for future in as_completed(futures):
                page_no = futures[future]
                try:
                    pages_content[page_no - 1] = future.result()
                except Exception as exc:
                    logging.error(
                        "[Row %s] Page %s metadata build failed: %s",
                        idx,
                        page_no,
                        exc,
                        exc_info=True,
                    )
                    pages_content[page_no - 1] = {
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
            "[Row %s] VLM extraction complete — total %.2fs",
            idx,
            time.time() - t_total_start,
        )
        return result

    except Exception as exc:
        logging.error(
            "[Row %s] VLM Docling extraction failed for %s: %s",
            idx,
            pdf_path,
            exc,
            exc_info=True,
        )
        return default_result(str(exc))


def do_docling_extraction(
    df: pd.DataFrame,
    *,
    api_url: str,
    model_name: str,
    api_timeout: float,
    api_concurrency: int,
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
        "Starting multiprocessing VLM extraction on %d records using max_workers=%d",
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
            api_url=api_url,
            model_name=model_name,
            api_timeout=api_timeout,
            api_concurrency=api_concurrency,
            debug_output_dir=debug_output_dir,
            log_file_path=log_file_path,
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

            if not in_flight:
                continue

            done, _ = wait(
                in_flight.keys(),
                timeout=poll_interval_seconds,
                return_when=FIRST_COMPLETED,
            )

            now = clock_fn()
            timed_out = [
                future
                for future, (idx, start_time) in in_flight.items()
                if (now - start_time) > task_timeout_seconds
            ]

            if timed_out:
                timed_out_rows = [in_flight[future][0] for future in timed_out]
                logging.error(
                    "[MainProc] Timed out rows=%s after %.1fs; restarting executor",
                    timed_out_rows,
                    task_timeout_seconds,
                )
                for future in list(in_flight.keys()):
                    idx, _start = in_flight.pop(future)
                    if future in timed_out:
                        mark_timeout(
                            df,
                            idx,
                            str(df.at[idx, "PDFPath"]),
                            str(df.at[idx, "FileName"]),
                        )
                        dirty_count += 1
                    else:
                        pending.insert(0, idx)
                shutdown_executor(executor, kill_workers=True)
                executor = create_executor(
                    max_workers=max_workers,
                    api_url=api_url,
                    model_name=model_name,
                    api_timeout=api_timeout,
                    api_concurrency=api_concurrency,
                    debug_output_dir=debug_output_dir,
                    log_file_path=log_file_path,
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
                continue

            for future in done:
                idx, _start = in_flight.pop(future)
                try:
                    raw_result = future.result()
                except Exception as exc:
                    logging.error(
                        "[MainProc] Future failed for row %s: %s",
                        idx,
                        exc,
                        exc_info=True,
                    )
                    raw_result = default_result(str(exc))

                result = validate_output_row(
                    str(df.at[idx, "PDFPath"]),
                    str(df.at[idx, "FileName"]),
                    raw_result,
                )
                assign_row_result(df, idx, result)
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

    finally:
        shutdown_executor(executor, kill_workers=False)
        if checkpoint_path:
            write_checkpoint(df, checkpoint_path)

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract full-PDF outputs by rendering pages, sending them to a local VLM API, and loading returned doctags into Docling."
    )
    parser.add_argument("input_path", help="PDF file or directory containing PDFs")
    parser.add_argument("output_feather", help="Output feather path")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8013/v1/chat/completions",
        help="OpenAI-compatible VLM API endpoint",
    )
    parser.add_argument(
        "--model",
        default="ibm-granite/granite-docling-258M",
        help="Model name to send in API params",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=120.0,
        help="Per-request API timeout in seconds",
    )
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="Concurrent API requests per worker",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes; default is 10",
    )
    parser.add_argument(
        "--debug-output-dir",
        default=None,
        help="Optional debug output directory",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for a single pdf task before worker restart",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for worker completion",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Checkpoint after this many completed rows",
    )
    parser.add_argument(
        "--save-interval-seconds",
        type=float,
        default=60.0,
        help="Checkpoint at least this often while work is progressing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output feather if present",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry rows that already have non-timeout errors in the resume file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_feather).expanduser().resolve()
    debug_output_dir = (
        Path(args.debug_output_dir).expanduser().resolve()
        if args.debug_output_dir
        else None
    )

    max_workers = args.max_workers or 10
    max_workers = max(1, max_workers)

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / (
        f"fullpdf_vlm_api_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    logging.info("input_path=%s", input_path)
    logging.info("output_feather=%s", output_path)
    logging.info("mode=direct_api_doctags")
    logging.info("response_format=%s", VLM_RESPONSE_FORMAT)
    logging.info("prompt=%s", VLM_PROMPT)
    logging.info("api_url=%s", args.api_url)
    logging.info("model=%s", args.model)
    logging.info("api_timeout=%.1f", args.api_timeout)
    logging.info("api_concurrency=%s", args.api_concurrency)
    logging.info("max_workers=%s", max_workers)
    if debug_output_dir is not None:
        logging.info("debug_output_dir=%s", debug_output_dir)

    df = create_pdf_dataframe(input_path)
    out_df = do_docling_extraction(
        df,
        api_url=args.api_url,
        model_name=args.model,
        api_timeout=args.api_timeout,
        api_concurrency=args.api_concurrency,
        max_workers=max_workers,
        debug_output_dir=debug_output_dir,
        log_file_path=str(log_file_path),
        output_path=str(output_path),
        save_every=args.save_every,
        save_interval_seconds=args.save_interval_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )

    write_checkpoint(out_df, output_path)
    logging.info("wrote output feather to %s", output_path)


if __name__ == "__main__":
    main()
