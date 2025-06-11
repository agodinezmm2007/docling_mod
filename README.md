# Modified & Enhanced Docling (docling_mod)

This repository contains a modified and significantly enhanced version of IBM's Docling library, designed for high-throughput, robust analysis of scholarly articles. These modifications were developed to overcome specific limitations in the original library related to performance, accuracy, and stability, as detailed in my [technical report](https://agodinezmm2007.github.io/project_portfolio/05-technical-report.html#stage-4-content-extraction-via-document-layout-analysis).

The code here is primarily for testing and validation purposes.

## Key Features & Enhancements

This modified version includes several critical enhancements over the base Docling library:

- **Multi-GPU Parallelization:** A custom architecture using Python's `ProcessPoolExecutor` allows Docling to run as a distributed application across multiple GPUs, dramatically reducing processing time.
- **Efficient Formula Recognition:** Replaced the resource-intensive (30 GB VRAM) formula model with SmolDocling model, reducing VRAM requirements to 8 GB and making it usable on smaller GPUs (12 GB 4070 Super).
- **Advanced Layout Post-Processing:** Engineered rule-based heuristics to automatically correct common layout analysis errors, such as merging fragmented mathematical formulas and re-classifying entire pages that were misidentified as tables.
- **Stability Fixes:** Addressed persistent glyph-parsing errors by reverse-engineering and patching the core C-extension library, preventing crashes during large-scale runs.

## Installation

It is strongly recommended to use a dedicated virtual environment (like `venv` or `conda`) to avoid conflicts with existing packages.

1. **Create and activate a new virtual environment.**
2. **Install Base Docling:** First, install the original `docling` package and its dependencies via pip. This will ensure all base requirements are met.
3. **Replace with Modified Files:** Copy the folders from the `site-packages` directory in this repository into your virtual environment's `site-packages` folder, replacing the original `docling` and `docling_ibm_models` folders.
4. **Install Additional Dependencies:** You will likely need to install a few extra packages. The main ones are:

```bash
   pip install accelerate flash-attn
```

## How to Test

This repository includes scripts and sample data to test the stability and functionality of the modified pipeline.

### Test Data

The `data/` folder contains a subfolder with **190 academic journal articles** ready for processing.

### Test Scripts

The `scripts/` folder contains several scripts to run the pipeline:

- `scripts/docling_test_single_GPU.py` — Tests processing on a single GPU.  
- `scripts/docling_extract_formulas_mp_multi.py` — Tests processing across multiple GPUs.  
- `scripts/docling_test_nb.ipynb` — A Jupyter Notebook that provides a step-by-step walkthrough of the process. This is the easiest way to get started.

## Debugging

If you want to visualize the model's intermediate steps, you can uncomment lines in the source code:

- **To Save Masked Page Images:**  
  Uncomment the `Image.save()` line around here in [base_models.py](https://github.com/agodinezmm2007/docling_mod/blob/ea18bf4a42373318ed9d108c4ca8d597a19a1151/site-packages/docling/datamodel/base_models.py#L341). This shows the page with all non-formula content grayed out.

- **To Save Formula Snippets:**  
  Uncomment the `snippet.save()` line around here in [code_formula_predictor.py](https://github.com/agodinezmm2007/docling_mod/blob/ea18bf4a42373318ed9d108c4ca8d597a19a1151/site-packages/docling_ibm_models/code_formula_model/code_formula_predictor.py#L280). This saves the cropped image of each formula sent to the "SmolDocling" model.

## Known Issues & Limitations

- **Stability:** The system may crash when processing certain PDFs, particularly those containing "prompting", mainly from articles discussing LLMs. This is a known issue that requires further troubleshooting.  
- **Layout Errors:** Some pages are still occasionally misclassified as large tables. The post-processing heuristics catch many of these but not all.  
- **GPU Errors:** While multi-GPU processing has been tested successfully on two NVlinked Ampere RTX A6000 GPUs, it has caused CUDA errors on a specific model (a Chinese-market 4090D). This may be a hardware-specific issue as it does not occur on the A6000s or the 4070. 




