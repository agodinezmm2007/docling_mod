# layout_model.py

Local: site-packages/docling/models/layout_model.py

GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/models/layout_model.py

Local lines: 243 | GitHub lines: 243

## Summary

A single-line change that passes `page_number=page.page_no` as an additional keyword argument to the `LayoutPostprocessor` constructor. This supplies per-page identification to the postprocessor, which the modified `layout_postprocessor.py` uses for page-aware spatial indexing and debug logging.

## Added Imports

None

## Modified Functions/Methods

### LayoutModel.__call__ (line 148)
- At line 215-216, the `LayoutPostprocessor` instantiation gains the keyword argument `page_number=page.page_no`:
  - GitHub: `LayoutPostprocessor(page, clusters, self.options)`
  - Local: `LayoutPostprocessor(page, clusters, self.options, page_number=page.page_no)`

## New Functions/Methods

None

## Modified Variables/Constants

None

## Added Comments

None

## Removed Code

None
