"""
PDF invisible text analysis and removal.

Consolidates analysis (PyMuPDF spans, pikepdf content streams),
single-file cleaning (pikepdf content stream filtering, Ghostscript re-render),
and parallel batch processing into one module.

Usage:
    python pdf_cleaner.py analyze <input.pdf> [--method spans|stream] [--max-pages N]
    python pdf_cleaner.py clean <input.pdf> [output.pdf] [--method ghostscript|content_stream]
    python pdf_cleaner.py batch [input_dir] [output_dir] [--workers N] [--timeout N]

Batch runs the full pipeline per PDF: analyze (spans + stream) -> clean (content stream) -> clean (ghostscript).
Defaults: input=data/NEW_ETL_PDF, output=scripts/sample_pdfs_cleaned.
All activity logged to scripts/logs/pdf_cleaner.log.
"""

import argparse
import logging
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "NEW_ETL_PDF"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sample_pdfs_cleaned"

LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / "pdf_cleaner.log"

logger = logging.getLogger("pdf_cleaner")


def _setup_logging():
    """Configure logging with file handler. Safe to call multiple times (clears existing handlers)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(str(LOG_FILE), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(processName)s[%(process)d] - %(levelname)s - %(message)s'
    ))
    root.addHandler(fh)


def _worker_init():
    """Initialize logging in each worker process."""
    _setup_logging()


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze_spans(pdf_path, max_pages=5):
    """
    PyMuPDF span-level analysis. Detects white text (color >= 0xF0F0F0)
    and tiny text (size < 1.0) by inspecting each text span's properties.

    Returns list of dicts: {text, color, size, flags, page, white, tiny}
    """
    import fitz

    logger.info("analyze_spans: %s (max_pages=%d)", pdf_path, max_pages)
    doc = None
    try:
        doc = fitz.open(pdf_path)
        results = []

        for page_num in range(min(max_pages, len(doc))):
            try:
                page = doc[page_num]
                text_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            except Exception as e:
                logger.warning("analyze_spans: page %d error in %s: %s", page_num + 1, pdf_path, e)
                continue

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        color = span.get("color", 0)
                        size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        results.append({
                            "text": text[:80],
                            "color": color,
                            "size": size,
                            "flags": flags,
                            "page": page_num + 1,
                            "white": color >= 0xF0F0F0,
                            "tiny": size < 1.0,
                        })

        logger.info("analyze_spans complete: %d spans from %s", len(results), pdf_path)
        return results
    except Exception:
        logger.exception("analyze_spans failed for %s", pdf_path)
        raise
    finally:
        if doc is not None:
            doc.close()


def analyze_content_stream(pdf_path):
    """
    pikepdf operator-level analysis. Tracks render mode, font size,
    and font name for each text-showing operator.

    Returns list of dicts: {text, render_mode, font_size, font, page}
    """
    import pikepdf
    from pikepdf import Pdf, Operator

    logger.info("analyze_content_stream: %s", pdf_path)
    pdf = None
    try:
        pdf = Pdf.open(pdf_path)
        results = []

        for page_num, page in enumerate(pdf.pages):
            if "/Contents" not in page:
                continue

            try:
                parsed = pikepdf.parse_content_stream(page)
            except Exception as e:
                logger.warning("analyze_content_stream: page %d parse error in %s: %s",
                               page_num + 1, pdf_path, e)
                continue

            render_mode = 0
            font_size = 12
            font_name = None

            for operands, operator in parsed:
                if operator == Operator("Tr") and operands:
                    render_mode = int(operands[0])

                elif operator == Operator("Tf") and operands and len(operands) > 1:
                    font_name = str(operands[0])
                    font_size = float(operands[1])

                elif operator == Operator("Tj") and operands:
                    text = str(operands[0])
                    if text:
                        results.append({
                            "text": text[:80],
                            "render_mode": render_mode,
                            "font_size": font_size,
                            "font": font_name,
                            "page": page_num + 1,
                        })

                elif operator == Operator("TJ") and operands:
                    text_parts = [
                        str(x) for x in operands[0]
                        if isinstance(x, (str, bytes))
                    ]
                    text = "".join(text_parts)
                    if text:
                        results.append({
                            "text": text[:80],
                            "render_mode": render_mode,
                            "font_size": font_size,
                            "font": font_name,
                            "page": page_num + 1,
                        })

        logger.info("analyze_content_stream complete: %d text items from %s", len(results), pdf_path)
        return results
    except Exception:
        logger.exception("analyze_content_stream failed for %s", pdf_path)
        raise
    finally:
        if pdf is not None:
            pdf.close()


# ---------------------------------------------------------------------------
# single-file cleaning
# ---------------------------------------------------------------------------

def clean_content_stream(input_path, output_path=None):
    """
    Remove invisible text by filtering pikepdf content stream operators.

    Detects and removes:
    - Render mode 3 (invisible text)
    - Zero/near-zero font size (Tf and Tm operators)

    Returns stats dict with counts per category and total removed.
    """
    import pikepdf
    from pikepdf import Pdf, Operator

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.pdf"
    else:
        output_path = Path(output_path)

    logger.info("clean_content_stream: %s -> %s", input_path, output_path)

    pdf = None
    try:
        pdf = Pdf.open(input_path)

        stats = {
            "render_mode_3": 0,
            "zero_font_size": 0,
            "total_removed": 0,
        }

        for page_num, page in enumerate(pdf.pages):
            if "/Contents" not in page:
                continue

            try:
                parsed = pikepdf.parse_content_stream(page)
            except Exception as e:
                logger.warning("clean_content_stream: page %d parse error in %s: %s",
                               page_num + 1, input_path.name, e)
                continue

            filtered_ops = []
            page_removed = 0
            render_mode = 0
            font_size = 12
            zero_font_active = False
            skip_next_text = False

            for operands, operator in parsed:
                # render mode
                if operator == Operator("Tr"):
                    if operands and len(operands) > 0:
                        render_mode = int(operands[0])
                        if render_mode == 3:
                            skip_next_text = True

                # font size via Tf
                elif operator == Operator("Tf"):
                    if operands and len(operands) > 1:
                        font_size = float(operands[1])
                        if abs(font_size) < 0.1:
                            zero_font_active = True
                        else:
                            zero_font_active = False

                # font size via Tm matrix
                elif operator == Operator("Tm"):
                    if operands and len(operands) >= 6:
                        a = float(operands[0])
                        b = float(operands[1])
                        c = float(operands[2])
                        d = float(operands[3])
                        effective_size = max(abs(a), abs(b), abs(c), abs(d))
                        if effective_size < 0.1:
                            zero_font_active = True
                        else:
                            zero_font_active = False

                # text-showing operators
                elif operator in [
                    Operator("Tj"), Operator("TJ"),
                    Operator("'"), Operator('"'),
                ]:
                    if skip_next_text or zero_font_active:
                        stats["total_removed"] += 1
                        if skip_next_text:
                            stats["render_mode_3"] += 1
                        elif zero_font_active:
                            stats["zero_font_size"] += 1
                        page_removed += 1
                        skip_next_text = False
                        continue

                filtered_ops.append([operands, operator])

            if page_removed > 0:
                logger.debug("clean_content_stream: page %d of %s: removed %d text operators",
                             page_num + 1, input_path.name, page_removed)

            # write filtered content back as a single stream
            try:
                new_content = pikepdf.unparse_content_stream(filtered_ops)
                page.Contents = pdf.make_stream(new_content)
            except Exception as e:
                logger.error("clean_content_stream: failed to write page %d for %s: %s",
                             page_num + 1, input_path.name, e, exc_info=True)
                raise

        try:
            pdf.save(output_path)
        except Exception as e:
            logger.error("clean_content_stream: failed to save %s: %s", output_path, e, exc_info=True)
            raise

        logger.info("clean_content_stream complete: %s | render_mode_3=%d, zero_font=%d, total_removed=%d",
                     input_path.name, stats["render_mode_3"], stats["zero_font_size"],
                     stats["total_removed"])
        return stats
    except Exception:
        logger.exception("clean_content_stream failed for %s", input_path)
        raise
    finally:
        if pdf is not None:
            pdf.close()


def clean_ghostscript(input_path, output_path=None, timeout=120):
    """
    Re-render PDF through Ghostscript to strip invisible content.

    Returns (success: bool, error: str | None).
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.pdf"
    else:
        output_path = Path(output_path)

    logger.info("clean_ghostscript: %s -> %s (timeout=%d)", input_path, output_path, timeout)

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/prepress",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/FlateEncode",
        "-dGrayImageFilter=/FlateEncode",
        f"-sOutputFile={output_path}",
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            logger.info("clean_ghostscript success: %s", input_path.name)
            return True, None
        error_msg = result.stderr[:500]
        logger.error("clean_ghostscript failed (rc=%d): %s | %s", result.returncode, input_path.name, error_msg)
        return False, error_msg
    except subprocess.TimeoutExpired:
        logger.error("clean_ghostscript timeout after %ds: %s", timeout, input_path.name)
        return False, f"Timeout after {timeout} seconds"
    except FileNotFoundError:
        logger.error("clean_ghostscript: Ghostscript (gs) not found in PATH")
        return False, "Ghostscript not found. Install with: sudo apt-get install ghostscript"
    except Exception as e:
        logger.exception("clean_ghostscript error for %s", input_path.name)
        return False, str(e)[:500]


# ---------------------------------------------------------------------------
# batch processing
# ---------------------------------------------------------------------------

def _batch_worker(args):
    """
    Worker function: full pipeline per PDF.
    1. Analyze with both methods (spans + content stream) and log findings
    2. Clean with content stream filtering (surgical removal)
    """
    pdf_path, output_dir, timeout = args
    pdf_path = Path(pdf_path)
    output_path = Path(output_dir) / pdf_path.name
    file_start = time.time()

    result = {
        "analysis": {"spans": None, "stream": None},
        "cleaning": {"content_stream": None},
        "error": None,
    }

    try:
        # --- Phase 1: Analyze ---
        try:
            span_results = analyze_spans(pdf_path)
            white_count = sum(1 for r in span_results if r["white"])
            tiny_count = sum(1 for r in span_results if r["tiny"])
            result["analysis"]["spans"] = {
                "total_spans": len(span_results),
                "white_text": white_count,
                "tiny_text": tiny_count,
            }
            if white_count or tiny_count:
                logger.info("ANALYSIS %s: spans found %d white, %d tiny text out of %d spans",
                            pdf_path.name, white_count, tiny_count, len(span_results))
        except Exception as e:
            logger.warning("ANALYSIS %s: span analysis failed: %s", pdf_path.name, e)

        try:
            stream_results = analyze_content_stream(pdf_path)
            invisible_count = sum(1 for r in stream_results if r["render_mode"] == 3)
            tiny_font_count = sum(1 for r in stream_results if abs(r["font_size"]) < 0.5)
            result["analysis"]["stream"] = {
                "total_items": len(stream_results),
                "invisible_render_mode": invisible_count,
                "tiny_font": tiny_font_count,
            }
            if invisible_count or tiny_font_count:
                logger.info("ANALYSIS %s: stream found %d invisible (render mode 3), %d tiny font out of %d items",
                            pdf_path.name, invisible_count, tiny_font_count, len(stream_results))
        except Exception as e:
            logger.warning("ANALYSIS %s: content stream analysis failed: %s", pdf_path.name, e)

        # --- Phase 2: Clean with content stream filtering ---
        try:
            cs_stats = clean_content_stream(pdf_path, output_path)
            result["cleaning"]["content_stream"] = cs_stats
        except Exception as e:
            logger.error("CLEAN %s: content stream cleaning failed: %s", pdf_path.name, e)
            # if content stream cleaning fails, copy original to output
            try:
                import shutil
                shutil.copy2(pdf_path, output_path)
            except Exception as copy_err:
                logger.error("CLEAN %s: failed to copy original: %s",
                             pdf_path.name, copy_err)

        elapsed = time.time() - file_start
        logger.info("DONE %s in %.1fs | cs_removed=%d",
                     pdf_path.name, elapsed,
                     cs_stats["total_removed"] if result["cleaning"]["content_stream"] else 0)

        return str(pdf_path), True, result

    except Exception as e:
        logger.exception("WORKER CRASH for %s", pdf_path.name)
        result["error"] = str(e)[:500]
        return str(pdf_path), False, result


def batch_clean(input_dir, output_dir, max_workers=None, timeout=120):
    """
    Clean all PDFs in a directory using the pipeline:
    analyze (spans + stream) -> clean (content stream).

    Args:
        input_dir: Directory containing PDFs.
        output_dir: Output directory for cleaned PDFs.
        max_workers: Number of parallel workers (default: min(cpu_count, 8)).
        timeout: Per-file timeout in seconds (unused, kept for CLI compat).

    Returns:
        (success_count, failed_count, failed_files) where failed_files is
        a list of (path, error) tuples.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create output directory: %s", output_path)
        raise

    try:
        pdf_files = sorted(input_path.glob("*.pdf"))
    except OSError:
        logger.exception("Failed to list PDFs in %s", input_path)
        raise

    total = len(pdf_files)

    if total == 0:
        logger.warning("No PDF files found in %s", input_path)
        print(f"No PDF files found in {input_path}")
        return 0, 0, []

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    logger.info("=== BATCH START === input=%s, output=%s, workers=%d, files=%d",
                input_path, output_path, max_workers, total)
    print(f"Found {total} PDF files in {input_path}")
    print(f"Output: {output_path}")
    print(f"Pipeline: analyze (spans+stream) -> clean (content_stream)")
    print(f"Workers: {max_workers}\n")

    success_count = 0
    failed_count = 0
    failed_files = []
    aggregate_stats = {
        "render_mode_3": 0,
        "zero_font_size": 0,
        "total_removed": 0,
    }
    start_time = time.time()

    tasks = [(pdf, output_path, timeout) for pdf in pdf_files]

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
        futures = {executor.submit(_batch_worker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                pdf_path, success, result = future.result()
                if success:
                    success_count += 1
                    cs_stats = result.get("cleaning", {}).get("content_stream")
                    removed = cs_stats["total_removed"] if cs_stats else 0
                    logger.info("[%d/%d] OK %s (removed=%d)",
                                success_count + failed_count, total, Path(pdf_path).name,
                                removed)
                    print(f"[{success_count + failed_count}/{total}] OK  {Path(pdf_path).name} "
                          f"(removed={removed})")
                    if cs_stats:
                        for key in aggregate_stats:
                            aggregate_stats[key] += cs_stats.get(key, 0)
                else:
                    failed_count += 1
                    error_msg = result.get("error", "unknown error") if isinstance(result, dict) else str(result)
                    failed_files.append((pdf_path, error_msg))
                    logger.error("[%d/%d] ERR %s: %s", success_count + failed_count, total,
                                 Path(pdf_path).name, error_msg)
                    print(f"[{success_count + failed_count}/{total}] ERR {Path(pdf_path).name}: {error_msg}")
            except Exception as e:
                failed_count += 1
                failed_files.append((str(pdf_file), str(e)))
                logger.exception("[%d/%d] WORKER CRASH %s", success_count + failed_count, total,
                                 Path(pdf_file).name)
                print(f"[{success_count + failed_count}/{total}] ERR {Path(pdf_file).name}: {e}")

    elapsed = time.time() - start_time

    print(f"\nCompleted: {success_count} success, {failed_count} failed, {elapsed:.1f}s elapsed")
    logger.info("=== BATCH COMPLETE === %d success, %d failed, %.1fs elapsed (%.2fs/file avg)",
                success_count, failed_count, elapsed, elapsed / total if total else 0)

    if aggregate_stats["total_removed"] > 0:
        logger.info("Aggregate removal stats: render_mode_3=%d, zero_font=%d, total_removed=%d",
                     aggregate_stats["render_mode_3"], aggregate_stats["zero_font_size"],
                     aggregate_stats["total_removed"])
        print(f"\nAggregate: {aggregate_stats['total_removed']} invisible text operators removed "
              f"(render_mode_3={aggregate_stats['render_mode_3']}, zero_font={aggregate_stats['zero_font_size']})")

    if failed_files:
        logger.error("=== FAILED FILES (%d) ===", len(failed_files))
        for path, error in failed_files:
            logger.error("  %s: %s", Path(path).name, error)
        print(f"\nFailed files:")
        for path, error in failed_files[:10]:
            print(f"  {Path(path).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see {LOG_FILE} for full list)")

    return success_count, failed_count, failed_files


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _print_span_results(results):
    white = [r for r in results if r["white"]]
    tiny = [r for r in results if r["tiny"]]
    pages = sorted(set(r["page"] for r in results))

    print(f"Total text spans: {len(results)} across {len(pages)} page(s)")

    if white:
        print(f"\nWhite/near-white text: {len(white)}")
        for r in white[:10]:
            print(f"  Page {r['page']}, color 0x{r['color']:06X}: {r['text']}")

    if tiny:
        print(f"\nTiny text (size < 1.0): {len(tiny)}")
        for r in tiny[:10]:
            print(f"  Page {r['page']}, size {r['size']:.2f}: {r['text']}")

    if not white and not tiny:
        print("No invisible text detected.")


def _print_stream_results(results):
    invisible = [r for r in results if r["render_mode"] == 3]
    tiny = [r for r in results if abs(r["font_size"]) < 0.5]
    pages = sorted(set(r["page"] for r in results))

    print(f"Total text items: {len(results)} across {len(pages)} page(s)")

    if invisible:
        print(f"\nInvisible text (render mode 3): {len(invisible)}")
        for r in invisible[:10]:
            print(f"  Page {r['page']}: {r['text']}")

    if tiny:
        print(f"\nTiny text (size < 0.5): {len(tiny)}")
        for r in tiny[:10]:
            print(f"  Page {r['page']}, size {r['font_size']:.2f}: {r['text']}")

    if not invisible and not tiny:
        print("No invisible text detected.")


def main():
    _setup_logging()
    logger.info("pdf_cleaner invoked: %s", sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="PDF invisible text analysis and removal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- analyze --
    p_analyze = sub.add_parser("analyze", help="Analyze a PDF for invisible text")
    p_analyze.add_argument("input", help="Input PDF file")
    p_analyze.add_argument(
        "--method", choices=["spans", "stream"], default="spans",
        help="Analysis method: spans (PyMuPDF) or stream (pikepdf). Default: spans",
    )
    p_analyze.add_argument(
        "--max-pages", type=int, default=5,
        help="Max pages to analyze (spans method only). Default: 5",
    )

    # -- clean --
    p_clean = sub.add_parser("clean", help="Clean invisible text from a single PDF")
    p_clean.add_argument("input", help="Input PDF file")
    p_clean.add_argument("output", nargs="?", default=None, help="Output PDF file (default: {stem}_cleaned.pdf)")
    p_clean.add_argument(
        "--method", choices=["ghostscript", "content_stream"], default="ghostscript",
        help="Cleaning method. Default: ghostscript",
    )
    p_clean.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (ghostscript only). Default: 120")

    # -- batch --
    p_batch = sub.add_parser("batch", help="Full pipeline: analyze + clean all PDFs in a directory")
    p_batch.add_argument("input_dir", nargs="?", default=str(DEFAULT_INPUT_DIR), help=f"Input directory containing PDFs (default: {DEFAULT_INPUT_DIR})")
    p_batch.add_argument("output_dir", nargs="?", default=str(DEFAULT_OUTPUT_DIR), help=f"Output directory for cleaned PDFs (default: {DEFAULT_OUTPUT_DIR})")
    p_batch.add_argument("--workers", type=int, default=None, help="Number of parallel workers. Default: min(cpu_count, 8)")
    p_batch.add_argument("--timeout", type=int, default=120, help="Per-file timeout in seconds (ghostscript). Default: 120")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "analyze":
            logger.info("Command: analyze, input=%s, method=%s", args.input, args.method)
            print(f"Analyzing: {args.input}")
            print(f"Method: {args.method}\n")
            if args.method == "spans":
                results = analyze_spans(args.input, max_pages=args.max_pages)
                _print_span_results(results)
            else:
                results = analyze_content_stream(args.input)
                _print_stream_results(results)

        elif args.command == "clean":
            logger.info("Command: clean, input=%s, output=%s, method=%s", args.input, args.output, args.method)
            print(f"Input:  {args.input}")
            print(f"Method: {args.method}")
            if args.method == "content_stream":
                stats = clean_content_stream(args.input, args.output)
                print(f"\nRemoval statistics:")
                print(f"  Render mode 3: {stats['render_mode_3']}")
                print(f"  Zero font size: {stats['zero_font_size']}")
                print(f"  White text: {stats['white_text']}")
                print(f"  Total removed: {stats['total_removed']}")
            else:
                success, error = clean_ghostscript(args.input, args.output, timeout=args.timeout)
                if success:
                    output = args.output or f"{Path(args.input).stem}_cleaned.pdf"
                    print(f"Cleaned PDF saved to: {output}")
                else:
                    print(f"Error: {error}")
                    sys.exit(1)

        elif args.command == "batch":
            logger.info("Command: batch, input_dir=%s, output_dir=%s, workers=%s",
                        args.input_dir, args.output_dir, args.workers)
            batch_clean(
                args.input_dir,
                args.output_dir,
                max_workers=args.workers,
                timeout=args.timeout,
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\nInterrupted.")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled exception")
        print(f"Fatal error: {e}")
        sys.exit(1)

    logger.info("pdf_cleaner finished")


if __name__ == "__main__":
    main()
