# Modified Docling (docling_mod)

The code here is for testing and validation.

Base versions: docling 2.62.0, docling_core 2.51.1, docling_ibm_models 3.10.2, docling_parse 4.7.1

Modified fork of IBM's Docling library for batch processing of academic PDFs. Details in the [technical report](https://agodinezmm2007.github.io/project_portfolio/05-technical-report.html#stage-4-content-extraction-via-document-layout-analysis).

## Glyphs

this version for docling 2.62.0 does not modify docling_parse/pdf_resources_v2/glyphs/standard/glyphlist.dat like in the previous version. common missing glyphs in the 198 test PDFs are

- /uniFB00 (ff ligature, 547 occurrences across 198 PDFs)
- /uniFB01 (fi ligature, 1180 occurrences)
- /uniFB02 (fl ligature, 298 occurrences)
- /uniFB03 (ffi ligature, 84 occurrences)
- /uniFB04 (ffl ligature, 1 occurrence)
- /uni03F5 (greek lunate epsilon, 4 occurrences)
- /uni262F (yin yang, 3 occurrences)
- /uni202F (narrow no-break space, 1 occurrence)

GLYPH<N> tokens from unmapped character codes (found in MDPI, Copernicus, PeerJ, PLOS, Elsevier PDFs):

- GLYPH<0>: 743 occurrences
- GLYPH<1>: 365
- GLYPH<21>: 147
- GLYPH<14>: 108
- GLYPH<25>: 63
- GLYPH<26>: 63
- GLYPH<6>: 60
- GLYPH<11>: 47
- GLYPH<8>: 28
- GLYPH<2>: 25
- GLYPH<15>: 22
- GLYPH<24>: 14
- GLYPH<c=0,font=/DejaVuMathTeXGyre-Regular>: 10
- GLYPH<13>: 8
- GLYPH<3>: 8
- GLYPH<12>: 3
- GLYPH<229>: 1

45 of 198 articles contain at least one GLYPH or /uni token. 22 have GLYPH<N> tokens (MDPI: 13, Copernicus: 5, PeerJ: 2, PLOS: 1, Elsevier: 1). 23 have /uni tokens (Elsevier: 8, BMJ: 5, Springer Nature: 2, Japan Epidemiological Association: 2, PLOS: 3, others: 3). No article has both types.

To attempt to resolve glyphs I attempt post processing instead of adding missing glyphs to source .dat files. To first detect glyphs I run `scripts/glyph_unifb_analysis.py` which detects instances of `uniFB` and `GLYPH<N>` in the FullText/text items on PagesJson columns. then ones theyre identified I attempt to resolve them using `scripts/fix_uni_tokens.py`. resolving `GLYPH<N>` requires to first map the `GLYPH<N>`s to their corresponding `uniFB` symbols

## Purpose

The primary purpose of this repository is to maximize throughput for batch PDF extraction. Using `scripts/docling_extract_formulas_mp_multi_provenance.py` on 2 GPUs with 10 workers, 198 academic PDFs (3,773 pages total) were fully processed in 6 minutes flat. That includes layout detection, formula/code extraction, table structure, markdown export, and page-level provenance assembly.

| Metric | Value |
| --- | --- |
| Wall clock time | 6.0 min |
| PDFs processed | 198 (0 errors) |
| Total pages | 3,773 |
| Avg time per PDF | 16.5s |
| Median time per PDF | 10.2s |
| Avg time per page | 0.86s |
| Throughput | 32.9 PDFs/min, 627.5 pages/min |

Hardware is as follows:

- 512 GB DDR4 ECC RDIMM ram
- Threadripper pro 5955wx
- GPU 0: NVIDIA RTX PRO 60000 max-q (96 gb)
- GPU 2: NVIDIA RTX A6000 (48 gb)
- GPU 3: NVIDIA RTX PRO 60000 max-q (96 gb)
- GPU 4: NVIDIA RTX A6000 (48 gb)

Minimum hardware is two GPUs: one for layout detection / table structure / formula extraction, the other running the vLLM docker container for code formula inference via the granite-vision API.

5 workers per A6000 (48 GB) is the maximum before CUDA OOM. Larger GPUs can run more.

### Worker/thread counts in `docling_extract_formulas_mp_multi_provenance.py`

| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| `max_workers` | `do_docling_extraction()` / `__main__` | `len(GPU_IDS) * 5` | Number of parallel PDF conversion processes across GPUs |
| `num_threads` | `AcceleratorOptions` in `init_pipeline()` | `6` | CPU threads per worker for model inference (layout, table structure) |
| `POST_PROCESS_WORKERS` | Module constant | `12` | Threads for table export and page metadata building within each PDF |
| `ThreadPoolExecutor` in table export | `extract_pdf_with_docling()` | `min(POST_PROCESS_WORKERS, num_tables)` | Capped to actual table count |

## Modifications

the following files were modified:

- `v2.62.0/docling/datamodel/base_models.py`
- `v2.62.0/docling/models/code_formula_model.py`
- `v2.62.0/docling/datamodel/pipeline_options.py`
- `v2.62.0/docling/utils/layout_postprocessor.py`
- `v2.62.0/docling/utils/visualization.py`
- `v3.10.2/docling_ibm_models/code_formula_model/code_formula_predictor.py`
- `v3.10.2/docling_ibm_models/layoutmodel/layout_predictor.py`

For the specifics on the changes refer to the `/comparisons` folder. some have more changes than others and describing the changes here feels redundant.

## Throughput Methods

- Multi-GPU parallelization via `ProcessPoolExecutor` with round-robin GPU assignment
- Formula/code extraction rewritten to use a vLLM API endpoint serving `granite-vision-3.3-2b` (replaces the stock codeformula/smoldocling model)

Docker command for granite-vision-3.3b-2b

```
docker run --name docling-granite-vision \
  --gpus '"device=0,3"' \
  --privileged \
  --ipc=host \
  -p 8006:8000 \
  -e OMP_NUM_THREADS=6 \
  -e VLLM_USE_V1=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /mnt/c/Users/WSTATION/Desktop/docling_mods/model_cache:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:25.09-py3 \
  vllm serve ibm-granite/granite-vision-3.3-2b \
      --port 8000 \
	  --tensor-parallel-size 2 \
	  --gpu-memory-utilization 0.9 \
	  --trust-remote-code \
	  --max-model-len 16384 
      --limit-mm-per-prompt '{"image": 1}' \
      --dtype auto
```

- scripts/code_formula_model_vllm_api.py is what site-packages/docling/models/code_formula_model.py looks like
- Layout post-processing: merging fragmented formulas, re-classifying pages misidentified as tables

## Installation

These modifications target docling 2.62.0 specifically

Use a dedicated virtual environment (`venv` or `conda`).

1. Create and activate the environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy the folders from `site-packages/` in this repository into the environment's `site-packages/`, overwriting the stock files.

## Extraction Scripts

Extraction scripts live in `scripts/`. All follow the same pattern: scan a PDF folder, build a DataFrame, run multi-process extraction, save results as a feather file.

Provenance scripts output `PagesJson` (per-page content with bounding boxes, reference detection, token counts) and `EquationsJson` with bbox provenance. Non-provenance scripts output `FullText`, `TablesJson`, and `EquationsJson` (latex only, no bbox).

| Script | GPUs | Provenance | Layout images |
| --- | --- | --- | --- |
| `docling_extract_formulas_mp_multi_provenance.py` | Multi | Yes | No |
| `docling_extract_formulas_debug_mp_mult_provenance.py` | Multi | Yes | Yes |
| `docling_extract_formulas_debug_mp_singl_provenance.py` | Single | Yes | Yes |
| `docling_extract_formulas_mp_multi.py` | Multi | No | Via `DEBUG_OUTPUT_PATH` |
| `docling_extract_formulas_mp_singl.py` | Single | No | Via `DEBUG_OUTPUT_PATH` |

## Other scripts

`scripts/pdf_cleaner.py` - used to strip characters in PDFs which mess with the pdf text parsing and layout detection. used at yoru own risk
`scripts/topic_text_reconstruction.py` - used to strip text of footnotes, section headers, references, author names, stray characters, and other text which could interfere with LDA topic modeling

### GPU Configuration

Edit the GPU variable near the top of each script (line 17):

```python
# multi-GPU scripts
GPU_IDS = [2, 4]

# single-GPU script
GPU_ID = 4
```

Set these to match the GPU indices on your machine (`nvidia-smi` to check).

### Running from the Command Line

```bash
cd scripts

# defaults: reads from sample_pdfs_cleaned/, writes to output/
python docling_extract_formulas_debug_mp_mult_provenance.py

# specify a PDF folder
python docling_extract_formulas_debug_mp_mult_provenance.py /path/to/pdfs

# specify PDF folder and output feather
python docling_extract_formulas_debug_mp_mult_provenance.py /path/to/pdfs /path/to/output.feather

# debug scripts accept a third arg for layout image output directory
python docling_extract_formulas_debug_mp_mult_provenance.py /path/to/pdfs /path/to/output.feather /path/to/debug_images
```

Each run creates a timestamped log file in `scripts/logs/` and a timestamped feather file in `scripts/output/`. Debug scripts also write raw and postprocessed layout images to the debug output directory (defaults to `docling_debug/` at the repo root).

### Running from the Notebook

`scripts/sample.ipynb` imports `do_docling_extraction` from whichever script you choose. To switch between scripts, change the import in cell 3:

```python
# non-debug (no layout images)
from docling_extract_formulas_mp_multi_provenance import do_docling_extraction

# debug (generates layout images)
from docling_extract_formulas_debug_mp_mult_provenance import do_docling_extraction
```

The `output_dir` parameter in cell 6 controls where debug layout images are saved. It has no effect when using the non-debug script.

### Test Data

`data/` contains a subfolder with 190 academic journal articles. `scripts/sample_pdfs/` has 12 PDFs for quick tests.

## Debugging

- Masked page images: uncomment the 4 lines at `base_models.py` line ~437 (`export_dir.mkdir(...)`, `export_file = ...`, `masked.save(...)`, `_log.info(...)`) in the venv's `site-packages/docling/datamodel/base_models.py`. Saves the page image with non-formula content masked out.
- Formula snippets: `code_formula_predictor.py` line ~182 (`img.save(image_filename, ...)`) in the venv's `site-packages/docling_ibm_models/code_formula_model/code_formula_predictor.py` saves snippet PNGs and DocTags HTML unconditionally during formula prediction. No uncommenting needed.

## Known Issues

- Some pages are still misclassified as large tables. The post-processing heuristics catch many but not all cases. lots of glyph issues 
