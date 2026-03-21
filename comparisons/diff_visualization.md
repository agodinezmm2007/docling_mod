# Modification Report: visualization.py

Local: site-packages/docling/utils/visualization.py
GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/utils/visualization.py
Local lines: 85 | GitHub lines: 85

## Summary

Two bug fixes correct Y-coordinate scaling that was using `scale_x` instead of `scale_y`, and one change to the cluster label text adds the cluster ID for debugging. Total line count is unchanged.

## Added Imports

None.

## Modified Functions/Methods

### `draw_clusters` (line 8)

Three single-line changes inside this function:

**Bug fix 1 (line 31):** Cell rectangle Y-coordinate scaling.
- Old: `cy0 *= scale_x`
- New: `cy0 *= scale_y`

The GitHub version scales `cy0` by `scale_x`, which is wrong: `cy0` is a Y-coordinate and should use `scale_y`. The local version corrects this. `cy1` was already correctly scaled by `scale_y` in both versions. This bug would cause distorted cell rectangles when `scale_x != scale_y`.

**Bug fix 2 (line 43):** Cluster bounding box Y-coordinate scaling.
- Old: `y0 *= scale_x`
- New: `y0 *= scale_y`

Same class of bug as fix 1, but for the cluster's own bounding box rather than its child cells. `y1` was already correctly scaled by `scale_y` in both versions.

**Label text change (line 62):**
- Old: `label_text = f"{c.label.name} ({c.confidence:.2f})"`
- New: `label_text = f"{c.label.name} (ID={c.id}, conf={c.confidence:.2f})"`

Adds the cluster ID to the drawn label overlay. The confidence value is prefixed with `conf=` for clarity. This makes debug visualization images show which cluster ID corresponds to each bounding box.

## New Functions/Methods

None.

## Modified Variables/Constants

None.

## Added Comments

None.

## Removed Code

None.
