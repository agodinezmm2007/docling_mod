# code_formula_predictor.py

Local: site-packages/docling_ibm_models/code_formula_model/code_formula_predictor.py

GitHub: https://github.com/docling-project/docling-ibm-models/blob/v3.10.2/docling_ibm_models/code_formula_model/code_formula_predictor.py

Local lines: 351 | GitHub lines: 290

## Summary

The file replaces IBM's SamOPT vision-language model (a SAM-based OPT causal LM used for code/formula recognition) with SmolDocling-256M-preview, an Idefics3-based model from `ds4sd/SmolDocling-256M-preview`. The model loading pipeline switches from `AutoTokenizer` + `SamOPTForCausalLM` + `SamOptImageProcessor` to `AutoProcessor` + `AutoModelForVision2Seq`. The stopping criteria class is replaced from `StopOnString` (token-ID matching against hardcoded LaTeX repetition patterns) to `RegexRepetitionStoppingCriteria` (regex-based detection of any repeating character sequence at the end of decoded text). The prompt construction changes from a hand-built `<img><imgpad>...</img>` chat template with special tokens (`<code_image_to_text>`, `<equation>`) to plain English prompts processed through `AutoProcessor.apply_chat_template()`. The `predict` method changes from batched inference (all images tokenized and generated together, with CPU/GPU code paths and `torch.autocast`) to sequential per-snippet inference (one `model.generate` call per image, with CUDA cache clearing between calls). A new `_parse_docling_output` method parses the raw SmolDocling output through `DocTagsDocument` and `DoclingDocument` to extract structured code/formula text, replacing the old `_strip` method that removed trailing LaTeX junk. A debugging helper `_export_snippet_doctags_html` is added but commented out at the call site.

## Added Imports

- `Iterable` from `typing` (line 5)
- `Path` from `pathlib` (line 9)
- `re` (lines 10, 26)
- `AutoProcessor`, `AutoModelForVision2Seq` from `transformers` (line 15)
- `DocTagsDocument` from `docling_core.types.doc.document` (line 16)
- `DoclingDocument` from `docling_core.types.doc` (line 17)
- `DocItemLabel` from `docling_core.types.doc.labels` (line 18)

## Removed Imports

- `AutoTokenizer` from `transformers` (GitHub line 12)
- `SamOPTForCausalLM` from `docling_ibm_models.code_formula_model.models.sam_opt` (GitHub line 14)
- `SamOptImageProcessor` from `docling_ibm_models.code_formula_model.models.sam_opt_image_processor` (GitHub lines 15-17)

## Modified Classes

### StopOnString -> RegexRepetitionStoppingCriteria (line 28)

The `StopOnString` class (GitHub line 25) is removed entirely. It accepted a `tokenizer` and `stop_string`, pre-encoded the stop string to token IDs, then checked every position in every sequence for that exact token subsequence. Five instances were created in `predict` to catch repetitive LaTeX patterns like `\quad \quad \quad \quad` and `\\ \\ \\ \\`.

Replaced by `RegexRepetitionStoppingCriteria` (line 28), which takes `tokenizer`, `repetition_threshold` (default 3), and `min_repeat_pattern` (default 4). On each `__call__`, it decodes the full output text and runs a regex `(.{N,}?)\1{M,}$` against the end of the string, where N = `min_repeat_pattern` and M = `repetition_threshold - 1`. If a match is found, it logs a warning with the repeating pattern and returns `True` to stop generation. This is a general-purpose approach that catches any repeating sequence of 4+ characters appearing 3+ times consecutively, rather than relying on hardcoded LaTeX-specific stop strings.

### CodeFormulaPredictor (line 48)

Class docstring replaced. GitHub version had a multi-line docstring documenting attributes (`_device`, `_num_threads`, `_tokenizer`, `_model`, `_image_processor`, `_temperature`). Local version states: "Modified to use ds4sd/SmolDocling-256M-preview (Idefics3)."

## Modified Functions/Methods

### __init__ (line 57)

- GitHub version: sets `self._device`, calls `torch.set_num_threads` if CPU, then inside `_model_init_lock` loads `AutoTokenizer.from_pretrained(artifacts_path, use_fast=True, padding_side="left")` into `self._tokenizer`, loads `SamOPTForCausalLM.from_pretrained(artifacts_path, device_map=self._device)` into `self._model`, calls `self._model.eval()`, and loads `SamOptImageProcessor.from_pretrained(artifacts_path)` into `self._image_processor`. Finishes with a debug log.
- Local version: sets `self._device`, calls `torch.set_num_threads` if CPU, logs "Loading SmolDocling from..." at INFO level (line 68). Then determines `attn_impl` and `dtype` based on `torch.cuda.is_available()`: `flash_attention_2` + `bfloat16` on CUDA, `eager` + `float32` on CPU (lines 71-78). Overrides `self._device` based on CUDA availability regardless of the `device` parameter passed in. Logs the configuration at INFO (lines 81-84). Inside `_model_init_lock`, loads `AutoProcessor.from_pretrained(artifacts_path)` into `self._processor` (line 88), loads `AutoModelForVision2Seq.from_pretrained(artifacts_path, torch_dtype=dtype, device_map="cpu", _attn_implementation=attn_impl)` into `self._model` (lines 89-94), then explicitly calls `self._model.to(device)` to move the model to the target GPU (line 96), and `self._model.eval()` (line 97). After the lock, creates `self.stopping_criteria` as a `StoppingCriteriaList` containing one `RegexRepetitionStoppingCriteria` with `repetition_threshold=4` and `min_repeat_pattern=4` (lines 101-107).
- Key difference: the model is loaded to CPU first via `device_map="cpu"`, then moved to the specified device with `.to(device)`. This is a two-step approach to avoid CUDA initialization issues during multi-process spawning.
- The docstring is removed entirely.

### info (line 109)

- Functionally identical. Docstring removed. Body simplified from assigning to a local variable and returning it, to returning the dict literal directly.

### _get_prompt (line 115)

- GitHub version: returns `"<code_image_to_text>"` for code, `"<equation>"` for formula, raises `NotImplementedError` otherwise. The returned query was concatenated into a long chat-style prompt with `<img><imgpad>*256</img>` image tokens and `USER:/ASSISTANT:` framing.
- Local version: returns `"Convert code to text."` for code (line 123), `"Convert formula to LaTeX."` for formula (line 125), `"Convert this page to docling."` for anything else (line 127). No `NotImplementedError`. The prompt is later passed through `self._processor.apply_chat_template()` in `predict` (line 238), which handles the HuggingFace chat template formatting for Idefics3.
- Docstring rewritten.

### predict (line 191)

This is the largest single change in the file.

**GitHub version** (GitHub line 179, 111 lines):
- Signature: `predict(self, images, labels, temperature=0.0)`
- Validates temperature, sets `do_sample` based on it
- Validates `len(labels) == len(images)`
- Converts all images to RGB PIL, then stacks them into a single `images_tensor` via `self._image_processor`
- Builds all prompts, tokenizes them together with `self._tokenizer(prompts, padding=True, return_tensors="pt")`
- Creates 5 `StopOnString` stopping criteria
- Two code paths: CPU calls `self._model.generate()` with `input_ids`, `attention_mask`, `images`, and `max_new_tokens=4096-prompt_len`; GPU wraps the same call in `torch.autocast(device_type=self._device, dtype=torch.bfloat16)`
- Both paths use `use_cache=True` and `no_repeat_ngram_size=200`
- Batch-decodes all outputs, runs `_strip` on each, returns the list

**Local version** (line 191, 108 lines):
- Signature: `predict(self, images, labels, snippet_ids=None)` -- `temperature` parameter removed, `snippet_ids` parameter added
- Validates `snippet_ids` length if provided, validates `len(images) == len(labels)`
- Processes each image sequentially in a `for` loop (line 221):
  1. Determines `snippet_id` from `snippet_ids[i]` or generates `f"snippet_{i}"` (lines 223-226)
  2. Type-checks each image as `Image.Image` (line 229)
  3. Builds prompt via `_get_prompt(lbl)` then wraps in a conversation structure for `apply_chat_template` (lines 233-238)
  4. Calls `self._processor(text=final_prompt, images=[img], return_tensors="pt").to(self._device)` (lines 242-246)
  5. Records `prompt_len = inputs["input_ids"].shape[1]` (line 248)
  6. Calls `torch.cuda.empty_cache()` and sets `self._model.config.use_cache = False` (lines 256-257)
  7. Calls `self._model.generate(**inputs, max_new_tokens=400, stopping_criteria=self.stopping_criteria)` (lines 260-266)
  8. Trims prompt tokens: `output_ids = generated_ids[:, prompt_len:]` (line 270)
  9. Decodes with `self._processor.batch_decode(output_ids, skip_special_tokens=False)[0].lstrip()` (line 272)
  10. Counts tokens and logs at DEBUG (lines 275-276)
  11. If empty output, logs warning and appends `""` (lines 279-282)
  12. Otherwise calls `_parse_docling_output(raw_text, lbl, img)` and appends result (lines 289-290)
- Returns `results` list (line 293)
- Notable: `max_new_tokens` reduced from `4096 - prompt_len` to a fixed `400`. `no_repeat_ngram_size` removed. `use_cache` set to `False`. `do_sample` and `temperature` handling removed. `numpy.ndarray` input support removed (only PIL Image accepted).

## New Functions/Methods

### _export_snippet_doctags_html (line 132)
- Signature: `def _export_snippet_doctags_html(self, doctag_output: str, img: Image.Image, snippet_id: str, output_dir="/mnt/c/Users/WSTATION/Desktop/NEW_ETL/snippet_html")`
- Debugging helper that exports SmolDocling raw output as viewable HTML alongside the source snippet image. Creates a per-document subdirectory by splitting `snippet_id` at `"_item_"`. Builds a `DocTagsDocument` from the raw text and image, loads it into a `DoclingDocument`, then calls `doc.save_as_html()` and `img.save()`. Handles filename collisions with a counter. Wrapped in try/except. The call site in `predict` is commented out (lines 284-286).

### _parse_docling_output (line 301)
- Signature: `def _parse_docling_output(self, raw_text: str, label: str, snippet_img: Image.Image) -> str`
- Replaces the old `_strip` method. Three-step parser:
  1. Creates `DocTagsDocument.from_doctags_and_image_pairs([raw_text], [snippet_img])` (line 313)
  2. Creates `DoclingDocument(name="Document")` and calls `doc.load_from_doctags(doctags_doc)` (lines 321-323)
  3. Iterates `doc.texts`, collecting `.text` from items matching `DocItemLabel.CODE` or `DocItemLabel.FORMULA` based on the `label` parameter (lines 333-338)
- Returns `"MALFORMED_FORMULA"` if the result is empty or contains `"<formula><loc_"` (lines 345-347)
- Each step individually wrapped in try/except with `_log.error` and early return of `""` on failure

## Removed Functions/Methods

### _strip (GitHub line 154)
- Iteratively removed trailing substrings (`\quad`, `\\`, `\,`, ` c c c c`, ` l l l l l`) from generated text. These were SamOPT-specific artifacts. Replaced by `_parse_docling_output` which parses structured DocTags output instead of cleaning raw text.

## Modified Variables/Constants

- `self._tokenizer` -> `self._processor` (the processor combines tokenizer and image processor for Idefics3)
- `self._image_processor` removed (folded into `self._processor`)
- `self.stopping_criteria` is now a persistent instance attribute initialized in `__init__` (line 101), rather than being created fresh on each `predict` call as in the GitHub version (GitHub line 250)

## Added Comments

- `# Global lock for model initialization to prevent threading issues` retained (line 24)
- `# Regex pattern: detects any repeated short sequence (4+ chars), 3 or more times consecutively at end.` (line 37)
- `# Specify the attention implementation dynamically` (line 70)
- `# Explicitly move to GPU after initialization` (line 95)
- Numbered step comments throughout `predict`: `# 2) For each snippet`, `# 3) Prepare inputs`, `# 4) (Optional) disable cache or clear CUDA`, `# 5) Generate`, `# 6) Trim out the prompt tokens`, `# 7) decode`, `# 8) Optionally log token count`, `# 9) Export snippet doc-ling HTML for debugging`, `# 10) parse doc-ling => recognized text`
- `# PREDICTING DOC TAGS / BELOW IS A NEW FUNCTION...` block comment after the `return results` in `predict` (lines 294-298), appears to be a development note about work in progress

## Removed Code

- All SamOPT model infrastructure: `SamOPTForCausalLM`, `SamOptImageProcessor`, `AutoTokenizer` imports and their usage
- `StopOnString` class with its 5 hardcoded LaTeX repetition patterns
- `_strip` method
- `temperature` parameter and all `do_sample`/temperature validation logic
- Batched image tensor stacking (`torch.stack([self._image_processor(img) for img in images_tmp])`)
- Batched tokenization (`self._tokenizer(prompts, padding=True, return_tensors="pt")`)
- CPU/GPU branching in `predict` (separate `if self._device == "cpu":` code paths with `torch.autocast`)
- `numpy.ndarray` image input conversion (`Image.fromarray(image).convert("RGB")`)
- All parameter-level docstrings (Parameters, Returns, Raises sections)
