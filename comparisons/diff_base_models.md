# Modification Report: base_models.py

Local: site-packages/docling/datamodel/base_models.py
GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/datamodel/base_models.py
Local lines: 576 | GitHub lines: 480

## Summary

The local version adds a masked image system to the `Page` class. This system creates page images where all non-formula layout clusters are whited out, leaving only formula regions visible. The masked images are cached per scale and can be cropped to a bounding box. The PIL import is restructured from importing the `Image` class directly to importing the `Image` and `ImageDraw` modules, and a module-level logger is added.

## Added Imports
- `Dict`, `List`, `Tuple` added to `typing` imports (line 3). `List` and `Tuple` are not actually used in the file.
- `from pathlib import Path` (line 4). Used only in commented-out debug export code.
- `from PIL import Image, ImageDraw` replaces `from PIL.Image import Image` (line 20). `ImageDraw` is used in `_create_masked_image`.
- `import logging` (line 30) and module-level `_log = logging.getLogger(__name__)` (line 31).

## Modified Functions/Methods
### ItemAndImageEnrichmentElement.image type annotation (line 296)
- `image: Image` changed to `image: Image.Image`. Required because the PIL import changed from importing the class directly (`from PIL.Image import Image`) to importing the module (`from PIL import Image`). All other type annotations using bare `Image` (e.g., `_image_cache`, `get_image`, the `image` property) were left unchanged and now refer to the PIL `Image` module rather than the `Image` class. This is a latent type inconsistency but does not cause runtime errors because Pydantic's `arbitrary_types_allowed=True` is set.

## New Functions/Methods
### Page.get_masked_image (line 370)
- Signature: `def get_masked_image(self, scale: float = 1.0, cropbox: Optional[BoundingBox] = None, pdf_identifier: Optional[str] = None) -> Optional[Image]`
- Checks `_masked_image_cache` for a cached masked image at the given scale. On cache miss, calls `_create_masked_image`. If `cropbox` is provided, crops the cached masked image to that region using the same coordinate transform as `get_image`.

### Page._create_masked_image (line 385)
- Signature: `def _create_masked_image(self, scale: float, pdf_identifier: Optional[str] = None) -> Optional[Image]`
- Copies the original page image from `get_image(scale)`. Iterates `self.predictions.layout.clusters`. For every cluster whose label is not `DocItemLabel.FORMULA`, computes an expanded bounding box (4.5% vertical expansion top and bottom via `top_expansion_factor = 0.045` and `bottom_expansion_factor = 0.045`), scales it to pixel coordinates, and draws a white-filled rectangle over it using `ImageDraw.Draw.rectangle`. Returns `None` and logs an error if any exception occurs. The `pdf_identifier` parameter feeds into commented-out debug export logic (lines 431-440) that would save the masked image to disk.

### Page.masked_image (property, line 448)
- Signature: `@property def masked_image(self) -> Optional[Image]`
- Convenience property that calls `get_masked_image(scale=self._default_image_scale)`, mirroring the existing `image` property pattern.

## Modified Variables/Constants
### Page._masked_image_cache (line 319)
- New private field: `_masked_image_cache: Dict[str, Image.Image] = {}`
- The type annotation says `Dict[str, Image.Image]` (string keys), but the actual usage in `get_masked_image` uses `scale` (a float) as the key, matching `_image_cache`'s pattern. The comment `# <-- Cache key changed to string` and the `Dict[str, ...]` annotation are vestiges of an earlier design that used a composite string key combining scale and cropbox hash.

## Added Comments
- `# New masked image cache, mirroring existing image cache` and `# Use a string key combining scale and cropbox hash for uniqueness` (lines 317-318) describe the cache field's intent.
- `# **** Modified Masked Image Implementation ****` (line 366) and `# **** Corrected Masked Image Implementation (Simple and Robust) ****` (line 369) are section headers from iterative development.
- `# Mask non-formula clusters explicitly here with correct scaling` (line 405).
- Lines 431-440: commented-out debug code that would save masked images to `/mnt/c/Users/WSTATION/Desktop/NEW_ETL/docling_debug/`, gated on `pdf_identifier`.
- `#-----------------------------------------------------------` (line 428): residual separator.
- Section boundary markers `# *****...` at lines 320 and 451.

## Removed Code
- Trailing newline at end of file removed (file now ends without a final newline).
