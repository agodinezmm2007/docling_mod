# Modification Report: pipeline_options.py

Local: site-packages/docling/datamodel/pipeline_options.py
GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/datamodel/pipeline_options.py
Local lines: 404 | GitHub lines: 387

## Summary

The local version adds a commented-out alternative definition of `VlmPipelineOptions` (lines 310-326) immediately after the active `VlmPipelineOptions` class. This block sketches a version of the class with additional fields for code/formula enrichment and picture classification/description, using `HuggingFaceVlmOptions` and `smoldocling_vlm_conversion_options` as the default VLM backend. The active class definition is unchanged. The file also loses its trailing newline.

## Added Imports
- None

## Modified Functions/Methods
- None. The active `VlmPipelineOptions` class (lines 300-308) is identical to the GitHub version.

## New Functions/Methods
- None

## Modified Variables/Constants
- None

## Added Comments
- Lines 310-326: a fully commented-out alternative `VlmPipelineOptions` class body. It adds `do_code_enrichment: bool = False`, `do_formula_enrichment: bool = False`, `do_picture_classification: bool = False`, `do_picture_description: bool = False`, and changes the default `vlm_options` type/value from `Union[InlineVlmOptions, ApiVlmOptions]` defaulting to `GRANITEDOCLING_TRANSFORMERS` to `HuggingFaceVlmOptions` defaulting to `smoldocling_vlm_conversion_options`. This appears to be a reference sketch for extending the VLM pipeline to support SmolDocling-based formula enrichment alongside the standard VLM conversion, never activated.

## Removed Code
- Trailing newline at end of file removed.
