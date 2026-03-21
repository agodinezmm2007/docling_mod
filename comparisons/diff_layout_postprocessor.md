# Modification Report: layout_postprocessor.py

Local: site-packages/docling/utils/layout_postprocessor.py
GitHub: https://github.com/docling-project/docling/blob/v2.62.0/docling/utils/layout_postprocessor.py
Local lines: 972 | GitHub lines: 683

## Summary

The local version adds three new post-processing stages to the layout pipeline: (1) merging vertically adjacent FORMULA clusters using Union-Find with formula-number-aware heuristics, (2) reclassifying oversized table/form/index clusters to TEXT when they cover most of the page but contain too few cells (table decontamination), and (3) applying per-label padding when adjusting FORMULA bounding boxes. It also accepts a `page_number` parameter for page-scoped debug logging and contains a commented-out stub for line-number cluster filtering. The file grew from 683 lines to 972 lines; all new code lives in `LayoutPostprocessor`.

## Added Imports

- `from typing import Dict, List, Set, Tuple, Optional` (line 6)
- `import numpy as np` (line 7)
- `import re` (line 8)

`numpy` is used for `np.median` in `_merge_vertically_adjacent_formulas`. `re` is used in `_extract_formula_number`. `Dict, List, Set, Tuple` from `typing` are imported but unused; only `Optional` is used (return type of `_extract_formula_number`).

## Modified Functions/Methods

### `__init__` (line 200)

Signature changed from `(self, page, clusters, options)` to `(self, page, clusters, options, page_number)`. The new `page_number: int` parameter is stored as `self.page_number` (line 208) and used in debug log messages throughout the new methods.

### `postprocess` (line 225)

The original final-cluster assembly:

```python
final_clusters = self._sort_clusters(
    self.regular_clusters + self.special_clusters, mode="id"
)
```

is replaced with a three-step sequence:

1. Combine regular and special clusters into `combined_clusters` (line 249).
2. Call `self._merge_vertically_adjacent_formulas(combined_clusters)` (line 252).
3. Call `self._filter_tables_containing_page_footer(combined_clusters)` (line 259).
4. Sort the result via `self._sort_clusters(combined_clusters, mode="id")` (line 262).

The original two lines that combined and sorted are commented out (lines 253-256). Six lines of commented-out line-number filtering code are inserted between `_process_special_clusters()` and the wrapper-contained-ID removal (lines 230-235), calling a `_filter_line_number_clusters` method that does not exist in this file.

### `_adjust_cluster_bboxes` (line 904)

A new `elif` branch is added after the TABLE branch (line 926) to handle `DocItemLabel.FORMULA` clusters. Instead of snapping the bbox directly to the cells bbox, it applies asymmetric padding:

```python
top_padding = 2
bottom_padding = 1
cluster.bbox = BoundingBox(
    l=cells_bbox.l,
    t=max(cells_bbox.t - top_padding, 0),
    r=cells_bbox.r,
    b=min(cells_bbox.b + bottom_padding, self.page_size.height),
)
```

This expands the formula bbox 2px above and 1px below the cell extents while clamping to page boundaries. No horizontal padding is applied.

## New Functions/Methods

### `_filter_tables_containing_page_footer` (line 289)

```python
def _filter_tables_containing_page_footer(self, clusters, min_area_ratio=0.70,
                                          min_cells_threshold=50,
                                          min_density_threshold=0.001) -> list[Cluster]
```

Iterates over clusters labelled TABLE, DOCUMENT_INDEX, KEY_VALUE_REGION, or FORM. For any cluster whose bounding box covers >= 70% of the page area, it collects all cells (recursively via `_collect_all_cells`) and computes cell density (cell count / bbox area). If the cluster has fewer than 50 cells or density below 0.001, it reclassifies the cluster to `DocItemLabel.TEXT`. Logs each decision with page number, cluster ID, area ratio, cell count, and density. Returns the mutated cluster list.

### `_collect_all_cells` (line 330)

```python
def _collect_all_cells(self, cluster: Cluster) -> list[TextCell]
```

Recursively collects cells from a cluster and all of its children. Uses `getattr(cluster, 'children', [])` for safe access. Called by `_filter_tables_containing_page_footer`.

### `_extract_formula_number` (line 341)

```python
def _extract_formula_number(self, cluster: Cluster) -> Optional[str]
```

Scans all cell texts in a cluster for parenthesized alphanumeric tokens matching `\(\s*([A-Za-z0-9]+)\s*\)`. Returns the first match with length 1-4 characters, or `None`. Used by `_merge_vertically_adjacent_formulas` to detect formula equation numbers like `(1)`, `(A1)`, `(10b)`. Contains a redundant `import re` inside the method body (re is also imported at module level).

### `_merge_vertically_adjacent_formulas` (line 360)

```python
def _merge_vertically_adjacent_formulas(self, clusters,
    vertical_threshold_factor=1.8, horizontal_overlap_threshold=0.7,
    padding=50, alignment_threshold=20, max_alignment_ratio=0.2)
```

The largest new method (~170 lines). Merges FORMULA clusters that are vertically adjacent on the page, using the existing `UnionFind` class. The algorithm:

1. Separates formula clusters from non-formula clusters (lines 393-394).
2. Sorts formula clusters by top coordinate (line 399).
3. Computes a dynamic vertical threshold as `median_height * 1.8` using `np.median` (lines 400-402).
4. Iterates all pairs `(i, j)` where `j > i`. For each pair, computes vertical gap, horizontal overlap (with 50px horizontal padding expansion on each side), left/right boundary differences, and a normalized alignment factor (lines 417-452).
5. Extracts formula numbers from both clusters via `_extract_formula_number` (lines 455-456).
6. Applies four branching merge rules based on formula number presence:
   - **Both have numbers, numbers differ**: skip merge (line 462).
   - **Both have numbers, numbers match**: use geometry-based `required_overlap` from the gap/alignment logic below (line 506).
   - **Neither has a number**: require vertical gap <= 3px and overlap >= 0.9 (lines 486-493).
   - **Exactly one has a number**: require vertical gap <= 12.8px and overlap >= 0.95 (lines 497-504).
7. Geometry-based `required_overlap` tiers (lines 269-281 of diff, lines 470-481 of local):
   - Gap <= 50% of threshold and edges aligned within 20px: use `horizontal_overlap_threshold` (0.7).
   - Gap <= 50% of threshold but edges not aligned: require 0.85.
   - Gap <= threshold and alignment factor <= 0.2: require 0.85.
   - Gap > threshold: skip.
8. Merges via `uf.union(c1.id, c2.id)` when overlap ratio meets the required threshold (line 511).
9. After union-find grouping, builds merged clusters with bounding boxes spanning all group members and deduplicated, sorted cells (lines 513-530).
10. Returns `non_formula_clusters + merged_formula_clusters` (line 531).

Contains a nested function `compute_alignment_factor(c1, c2)` (line 383) that computes the maximum of `left_diff / avg_width` and `right_diff / avg_width`. Contains a redundant `import numpy as np` inside the method body.

### Commented-out `_has_table_label` (lines 279-287)

A fully commented-out method that would recursively check if a cluster or its children have a table-like label. Not called anywhere.

## Modified Variables/Constants

- `# modified version` comment added at line 1 as a file-level marker.
- The blank line between `from collections import defaultdict` and `from docling_core...` (GitHub line 5) was removed; the new typing/numpy/re imports occupy that space.
- Trailing newline removed from end of file (the GitHub version ends with `\n`, the local version does not).

## Added Comments

- `# modified version` at line 1 marks the file as patched.
- `#--------------------------------------------------------------` separator comments at lines 247, 278, 339 delineate the new code blocks.
- `# new` inline comment at line 249 on the combined_clusters assignment.
- Commented-out `print()` debug statements throughout `_merge_vertically_adjacent_formulas` (approximately 12 instances), each paired with an active `_log.debug()` call. These are development artifacts.
- Inline comments on padding values in `_adjust_cluster_bboxes` explaining the purpose of each padding constant (lines 928-934).

## Removed Code

- The original three-line final cluster assembly in `postprocess` (GitHub lines 237-240) is commented out and replaced with the new merge/filter/sort sequence.
- The bare `else:` branch in `_adjust_cluster_bboxes` (GitHub line 648) that assigned `cluster.bbox = cells_bbox` for all non-TABLE clusters is narrowed: FORMULA clusters now take the new `elif` branch instead of falling through to `else`.
