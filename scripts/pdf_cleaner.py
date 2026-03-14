#!/usr/bin/env python3
"""
PDF invisible text analysis and removal.

Consolidates analysis (PyMuPDF spans, pikepdf content streams),
single-file cleaning (pikepdf content stream filtering, Ghostscript re-render),
and parallel batch processing into one module.

Usage:
    python pdf_cleaner.py analyze <input.pdf> [--method spans|stream] [--max-pages N]
    python pdf_cleaner.py clean <input.pdf> [output.pdf] [--method ghostscript|content_stream]
    python pdf_cleaner.py batch <input_dir> <output_dir> [--method ghostscript|content_stream] [--workers N] [--timeout N]
"""

import argparse
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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

    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        text_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)

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

    doc.close()
    return results


def analyze_content_stream(pdf_path):
    """
    pikepdf operator-level analysis. Tracks render mode, font size,
    and font name for each text-showing operator.

    Returns list of dicts: {text, render_mode, font_size, font, page}
    """
    import pikepdf
    from pikepdf import Pdf, Operator

    pdf = Pdf.open(pdf_path)
    results = []

    for page_num, page in enumerate(pdf.pages):
        if "/Contents" not in page:
            continue

        contents = page.Contents
        if not isinstance(contents, list):
            contents = [contents]

        for content in contents:
            try:
                parsed = pikepdf.parse_content_stream(content)
            except Exception:
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

    pdf.close()
    return results


# ---------------------------------------------------------------------------
# single-file cleaning
# ---------------------------------------------------------------------------

def clean_content_stream(input_path, output_path=None):
    """
    Remove invisible text by filtering pikepdf content stream operators.

    Detects and removes:
    - Render mode 3 (invisible text)
    - Zero/near-zero font size (Tf and Tm operators)
    - Clipped text (W/W* operators)
    - White text (g/rg color operators with value > 0.99)

    Returns stats dict with counts per category and total removed.
    """
    import pikepdf
    from pikepdf import Pdf, Operator, Array

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.pdf"
    else:
        output_path = Path(output_path)

    pdf = Pdf.open(input_path)

    stats = {
        "render_mode_3": 0,
        "zero_font_size": 0,
        "clipped": 0,
        "white_text": 0,
        "total_removed": 0,
    }

    for page_num, page in enumerate(pdf.pages):
        if "/Contents" not in page:
            continue

        contents = page.Contents
        if not isinstance(contents, list):
            contents = [contents]

        content_streams = []
        for content in contents:
            try:
                parsed = pikepdf.parse_content_stream(content)
            except Exception as e:
                print(f"Warning: page {page_num + 1} content stream error: {e}")
                content_streams.append(content)
                continue

            filtered_ops = []
            render_mode = 0
            font_size = 12
            in_clipping = False
            current_color = None
            skip_next_text = False

            for operands, operator in parsed:
                # render mode
                if operator == Operator("Tr"):
                    if operands and len(operands) > 0:
                        render_mode = int(operands[0])
                        if render_mode == 3:
                            skip_next_text = True
                            stats["render_mode_3"] += 1

                # font size via Tf
                elif operator == Operator("Tf"):
                    if operands and len(operands) > 1:
                        font_size = float(operands[1])
                        if abs(font_size) < 0.1:
                            skip_next_text = True
                            stats["zero_font_size"] += 1

                # font size via Tm matrix
                elif operator == Operator("Tm"):
                    if operands and len(operands) >= 6:
                        a = float(operands[0])
                        b = float(operands[1])
                        c = float(operands[2])
                        d = float(operands[3])
                        effective_size = max(abs(a), abs(b), abs(c), abs(d))
                        if effective_size < 0.1:
                            skip_next_text = True
                            stats["zero_font_size"] += 1

                # clipping paths
                elif operator in [Operator("W"), Operator("W*")]:
                    in_clipping = True
                    stats["clipped"] += 1

                # gray color
                elif operator == Operator("g"):
                    if operands and len(operands) > 0:
                        if float(operands[0]) > 0.99:
                            current_color = "white"

                # rgb color
                elif operator == Operator("rg"):
                    if operands and len(operands) >= 3:
                        r, g, b = float(operands[0]), float(operands[1]), float(operands[2])
                        if r > 0.99 and g > 0.99 and b > 0.99:
                            current_color = "white"
                            stats["white_text"] += 1

                # text-showing operators
                elif operator in [
                    Operator("Tj"), Operator("TJ"),
                    Operator("'"), Operator('"'),
                ]:
                    if skip_next_text or in_clipping or current_color == "white":
                        stats["total_removed"] += 1
                        skip_next_text = False
                        continue

                # save/restore graphics state
                elif operator == Operator("Q"):
                    in_clipping = False

                filtered_ops.append([operands, operator])

            new_content = pikepdf.unparse_content_stream(filtered_ops)
            content_streams.append(new_content)

        if len(content_streams) == 1:
            page.Contents = pdf.make_stream(content_streams[0])
        elif len(content_streams) > 1:
            page.Contents = Array([pdf.make_stream(cs) for cs in content_streams])

    pdf.save(output_path)
    pdf.close()

    return stats


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
            return True, None
        return False, result.stderr[:300]
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds"
    except FileNotFoundError:
        return False, "Ghostscript not found. Install with: sudo apt-get install ghostscript"
    except Exception as e:
        return False, str(e)[:300]


# ---------------------------------------------------------------------------
# batch processing
# ---------------------------------------------------------------------------

def _batch_worker_gs(args):
    """Worker function for parallel Ghostscript cleaning."""
    pdf_path, output_dir, timeout = args
    pdf_path = Path(pdf_path)
    output_path = Path(output_dir) / pdf_path.name
    success, error = clean_ghostscript(pdf_path, output_path, timeout)
    return str(pdf_path), success, error


def _batch_worker_cs(args):
    """Worker function for parallel content stream cleaning."""
    pdf_path, output_dir, _ = args
    pdf_path = Path(pdf_path)
    output_path = Path(output_dir) / pdf_path.name
    try:
        stats = clean_content_stream(pdf_path, output_path)
        return str(pdf_path), True, None
    except Exception as e:
        return str(pdf_path), False, str(e)[:300]


def batch_clean(input_dir, output_dir, method="ghostscript", max_workers=None, timeout=120):
    """
    Clean all PDFs in a directory using parallel processing.

    Args:
        input_dir: Directory containing PDFs.
        output_dir: Output directory for cleaned PDFs.
        method: "ghostscript" or "content_stream".
        max_workers: Number of parallel workers (default: min(cpu_count, 8)).
        timeout: Per-file timeout in seconds (Ghostscript only).

    Returns:
        (success_count, failed_count, failed_files) where failed_files is
        a list of (path, error) tuples.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    total = len(pdf_files)

    if total == 0:
        print(f"No PDF files found in {input_path}")
        return 0, 0, []

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    worker = _batch_worker_gs if method == "ghostscript" else _batch_worker_cs

    print(f"Found {total} PDF files in {input_path}")
    print(f"Output: {output_path}")
    print(f"Method: {method}, workers: {max_workers}\n")

    success_count = 0
    failed_count = 0
    failed_files = []

    tasks = [(pdf, output_path, timeout) for pdf in pdf_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                pdf_path, success, error = future.result()
                if success:
                    success_count += 1
                    print(f"[{success_count + failed_count}/{total}] OK  {Path(pdf_path).name}")
                else:
                    failed_count += 1
                    failed_files.append((pdf_path, error))
                    print(f"[{success_count + failed_count}/{total}] ERR {Path(pdf_path).name}: {error}")
            except Exception as e:
                failed_count += 1
                failed_files.append((str(pdf_file), str(e)))
                print(f"[{success_count + failed_count}/{total}] ERR {Path(pdf_file).name}: {e}")

    print(f"\nCompleted: {success_count} success, {failed_count} failed")

    if failed_files:
        print(f"\nFailed files:")
        for path, error in failed_files[:10]:
            print(f"  {Path(path).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

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
    p_batch = sub.add_parser("batch", help="Clean all PDFs in a directory")
    p_batch.add_argument("input_dir", help="Input directory containing PDFs")
    p_batch.add_argument("output_dir", help="Output directory for cleaned PDFs")
    p_batch.add_argument(
        "--method", choices=["ghostscript", "content_stream"], default="ghostscript",
        help="Cleaning method. Default: ghostscript",
    )
    p_batch.add_argument("--workers", type=int, default=None, help="Number of parallel workers. Default: min(cpu_count, 8)")
    p_batch.add_argument("--timeout", type=int, default=120, help="Per-file timeout in seconds (ghostscript only). Default: 120")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        print(f"Analyzing: {args.input}")
        print(f"Method: {args.method}\n")
        if args.method == "spans":
            results = analyze_spans(args.input, max_pages=args.max_pages)
            _print_span_results(results)
        else:
            results = analyze_content_stream(args.input)
            _print_stream_results(results)

    elif args.command == "clean":
        print(f"Input:  {args.input}")
        print(f"Method: {args.method}")
        if args.method == "content_stream":
            stats = clean_content_stream(args.input, args.output)
            print(f"\nRemoval statistics:")
            print(f"  Render mode 3: {stats['render_mode_3']}")
            print(f"  Zero font size: {stats['zero_font_size']}")
            print(f"  Clipped text: {stats['clipped']}")
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
        batch_clean(
            args.input_dir,
            args.output_dir,
            method=args.method,
            max_workers=args.workers,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
