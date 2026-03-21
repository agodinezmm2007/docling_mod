# Modification Report: code_formula_model.py

Local: site-packages/docling/models/code_formula_model.py
GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/models/code_formula_model.py
Local lines: 480 | GitHub lines: 337

## Summary

The file has been rewritten to replace local HuggingFace model inference (CodeFormulaV2 via `transformers.AutoModelForImageTextToText`) with remote inference through a vLLM API endpoint serving `ibm-granite/granite-vision-3.3-2b` on port 8006. The local version removes all model loading, tokenizer, and generation code, and instead sends base64-encoded snippet images to the API via HTTP POST. It adds asymmetric bounding box expansion with separate left/right factors, ratio-based image padding using edge-color detection, a DocTags-based output parser that constructs a `DocTagsDocument` to extract recognized text, and a debugging HTML exporter. The `_post_process` and `download_models` methods are deleted. The `__call__` method switches from batch GPU inference to a sequential per-snippet API call loop with per-element error handling.

## Added Imports

- `logging` (line 6)
- `base64` (line 7)
- `requests` (line 8)
- `BytesIO` from `io` (line 9)
- `Counter` from `collections` (line 10)
- `ImageOps` from `PIL` (line 25, added to existing `from PIL import Image`)
- `BoundingBox`, `DocItem` from `docling_core.types.doc` (lines 21-22)
- `ConversionResult` from `docling.datamodel.document` (line 30)
- `DocTagsDocument` from `docling_core.types.doc.document` (line 32)
- `Iterable` moved from `collections.abc` to `typing` (line 12)

## Removed Imports

- `AutoModelForImageTextToText`, `AutoProcessor` from `transformers` (GitHub line 17)
- `AcceleratorDevice` from `docling.datamodel.accelerator_options` (GitHub line 19)
- `download_hf_model` from `docling.models.utils.hf_model_download` (GitHub line 22)
- `decide_device` from `docling.utils.accelerator_utils` (GitHub line 23)

## Module-Level Additions

- `_log = logging.getLogger(__name__)` (line 34)
- File header comment block at lines 1-3: `# code_formula_model_vllm_api.py` / `# Modified to use VLLM API endpoint on port 8006 instead of local model` / `# Sends formula/code snippets to granite-vision-3.3-2b via API`

## Modified Functions/Methods

### CodeFormulaModelOptions (line 37)
- `kind` field changed from `Literal["code_formula"] = "code_formula"` to `Literal["vllm_api"] = "vllm_api"` (line 41)
- Two new fields added:
  - `api_url: str = "http://localhost:8006/v1/chat/completions"` (line 44)
  - `api_timeout: int = 120` (line 45)
- Docstring attributes section removed

### CodeFormulaModel.__init__ (line 60)
- Calls `super().__init__()` (line 71), which the GitHub version does not
- Stores `self.api_url = options.api_url` and `self.api_timeout = options.api_timeout` (lines 74-75)
- All local model initialization removed: no `decide_device`, no `AutoProcessor.from_pretrained`, no `AutoModelForImageTextToText.from_pretrained`, no `self._model.eval()`
- Logs initialization info via `_log.info` (lines 77-80)

### CodeFormulaModel.is_processable (line 82)
- Logic is identical but restructured: early `return False` if not enabled (line 87), then two separate `if` blocks with explicit `return True` (lines 91, 100), then `return False` (line 102). The GitHub version uses a single compound boolean return expression.

### CodeFormulaModel._extract_code_language (line 145)
- Logic unchanged. Docstring shortened. Inline comments on group(1)/group(2) removed.

### CodeFormulaModel._get_code_language_enum (line 159)
- Logic unchanged. Docstring shortened.

### CodeFormulaModel._get_prompt (line 171)
- GitHub version builds a HuggingFace chat template with `<code>` or `<formula>` tokens, calls `self._processor.apply_chat_template()`, and raises `NotImplementedError` for unknown labels
- Local version returns plain English prompt strings for the VLM API:
  - Code: `"Convert this code snippet to text. Identify the programming language and output the code."` (line 176)
  - Formula: `"Convert this mathematical formula to LaTeX. Output only the LaTeX code without explanations or delimiters."` (line 178)
  - Default fallback: `"Convert this image to text."` (line 180)
- Also accepts `DocItemLabel.CODE` and `DocItemLabel.FORMULA` enum values in addition to string labels (lines 175, 177)

### CodeFormulaModel.__call__ (line 390)
- GitHub version: collects all elements into lists, builds prompts, runs `self._processor()` to tokenize, calls `self._model.generate()` for batch GPU inference, calls `self._processor.batch_decode()`, runs `_post_process`, then yields items with assigned text
- Local version:
  - Collects elements with per-element error handling via try/except (line 407)
  - Filters non-TextItem elements by yielding them immediately (lines 409-411)
  - Applies `_pad_with_most_frequent_edge_color()` to each image (line 422)
  - Builds `snippet_ids` list as `f"{doc.name}_item_{idx}"` (line 429)
  - Processes each snippet sequentially via `_send_to_vllm_api()` (line 446)
  - Parses API responses through `_parse_docling_output()` (line 456)
  - Skips items returning `"MALFORMED_FORMULA"` or empty strings (lines 459-461)
  - Yields items individually inside the processing loop (line 474)
  - Each step wrapped in try/except with `_log.error` (lines 406, 476)

## New Functions/Methods

### prepare_element (line 104)
- Signature: `def prepare_element(self, conv_res: ConversionResult, element: NodeItem) -> Optional[ItemAndImageEnrichmentElement]`
- Overrides the base class method to implement asymmetric bounding box expansion. Computes `width` and `height` from the element's provenance bbox, then builds `expanded_bbox` using `left_expansion_factor` (0.04) for left, `right_expansion_factor` (0.03) for right, and `expansion_factor` (0.18) for top/bottom (lines 121-127).
- For formula elements, calls `page.get_masked_image()` with a `pdf_identifier` argument (lines 133-137). For code elements, calls `page.get_image()` (lines 139-141).
- Returns `ItemAndImageEnrichmentElement(item=element, image=cropped_image)` (line 143).

### _get_most_frequent_edge_color (line 184)
- Signature: `def _get_most_frequent_edge_color(self, pil_img: Image.Image) -> Union[int, Tuple[int, int, int]]`
- Extracts all pixel values along the four edges of an image using numpy slicing, counts them with `collections.Counter`, and returns the most common value. Handles grayscale (2D array, returns int) and color (3D array, returns RGB tuple) separately.

### _pad_with_most_frequent_edge_color (line 210)
- Signature: `def _pad_with_most_frequent_edge_color(self, img: Union[Image.Image, np.ndarray], pad_top_ratio: float = 0.13, pad_bottom_ratio: float = 0.13, pad_left_ratio: float = 0.03, pad_right_ratio: float = 0.03) -> Image.Image`
- Computes padding amounts as a ratio of image dimensions, determines border color via `_get_most_frequent_edge_color`, and applies the border using `PIL.ImageOps.expand()`.

### _export_snippet_doctags_html (line 238)
- Signature: `def _export_snippet_doctags_html(self, doctag_output: str, img: Image.Image, snippet_id: str, output_dir: str = "/mnt/c/Users/WSTATION/Desktop/NEW_ETL/snippet_html")`
- Debugging helper. Creates a per-document subdirectory, constructs a `DocTagsDocument` from the raw doctag output and image, loads it into a `DoclingDocument`, and saves both an HTML rendering and the snippet PNG. Handles filename collisions by appending a counter. Called in `__call__` but commented out (line 453).

### _parse_docling_output (line 284)
- Signature: `def _parse_docling_output(self, raw_text: str, label: str, snippet_img: Image.Image) -> str`
- Three-step parser:
  1. Creates `DocTagsDocument.from_doctags_and_image_pairs([raw_text], [snippet_img])` (line 294)
  2. Loads into `DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")` (line 303)
  3. Iterates `doc.texts`, collecting text from items whose label matches the expected label (code or formula) (lines 314-319)
- Returns `"MALFORMED_FORMULA"` if the result is empty or contains `"<formula><loc_"` (lines 326-328)
- Each step wrapped in try/except with `_log.error` and early return of `""` on failure

### _send_to_vllm_api (line 334)
- Signature: `def _send_to_vllm_api(self, img: Image.Image, prompt: str, snippet_id: str) -> Optional[str]`
- Converts PIL image to base64 PNG via `BytesIO` (lines 340-342)
- Builds OpenAI-compatible chat completion payload with model `"ibm-granite/granite-vision-3.3-2b"`, `max_tokens=2048`, `temperature=0.0` (lines 345-366)
- Sends via `requests.post()` with configurable timeout (lines 371-375)
- Returns `result['choices'][0]['message']['content']` on HTTP 200, `None` otherwise (lines 377-384)

## Modified Variables/Constants

- `_model_repo_folder = "docling-project--CodeFormulaV2"` removed (was GitHub line 68)
- `images_scale`: `1.67` -> `2.6` (line 55). Comment `# = 120 dpi, aligned with training data resolution` removed.
- `right_expansion_factor = 0.03` added (line 57)
- `left_expansion_factor = 0.04` added (line 58)

## Removed Code

### download_models (GitHub lines 117-129)
- Static method that called `download_hf_model(repo_id="docling-project/CodeFormulaV2", ...)`. Removed because the local version does no local model loading.

### _post_process (GitHub lines 247-275)
- Method that truncated text at `<end_of_utterance>` and stripped `</code>`, `</formula>`, and `<loc_0><loc_0><loc_500><loc_500>` tokens. Replaced by `_parse_docling_output` which parses DocTags structures instead of doing string cleanup.

### Local model inference in __call__ (GitHub lines 302-329)
- The entire block that collected labels/images into lists, called `self._processor(text=prompts, images=images, return_tensors="pt")`, moved inputs to device, set `gen_kwargs`, called `self._model.generate()`, and ran `self._processor.batch_decode()`. Replaced by per-snippet `_send_to_vllm_api` calls.

### Local model initialization in __init__ (GitHub lines 97-115)
- `decide_device()` call, `artifacts_path` resolution, `AutoProcessor.from_pretrained()`, `self._model_max_length`, `AutoModelForImageTextToText.from_pretrained()`, and `self._model.eval()`. All removed.

## Added Comments

- Header block (lines 1-3) documenting the vLLM API modification purpose
- `# Asymmetric bounding box expansion` (line 120)
- `# Use masked image for formulas, regular for code` (line 132)
- `# Apply ratio-based padding` (line 420)
- `# Optional: Export snippet for debugging` (line 452, preceding the commented-out `_export_snippet_doctags_html` call)
