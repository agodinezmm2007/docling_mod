# Docling Document API Reference

Source: `docling_core.types.doc`

Reference URL: https://docling-project.github.io/docling/reference/docling_document/

---

## Classes Summary

- `DoclingDocument` -- Primary document representation with full structure
- `DocumentOrigin` -- Source file metadata
- `DocItem` -- Base class for content-carrying elements
- `DocItemLabel` -- Enum for content types
- `ProvenanceItem` -- Extraction provenance tracking
- `GroupItem` -- Container for grouped content
- `GroupLabel` -- Enum for group types
- `NodeItem` -- Generic document tree node
- `PageItem` -- Page representation
- `FloatingItem` -- Base for floating elements (figures, tables)
- `TextItem` -- Text content element
- `TableItem` -- Table element
- `TableCell` -- Individual table cell
- `TableData` -- Table structure and cell data
- `TableCellLabel` -- Enum for table cell roles
- `KeyValueItem` -- Key-value pair element
- `SectionHeaderItem` -- Section heading element
- `PictureItem` -- Image/picture element
- `ImageRef` -- Image reference and metadata
- `PictureClassificationClass` -- Single classification result
- `PictureClassificationData` -- Picture classification metadata
- `RefItem` -- Cross-reference to another document item
- `BoundingBox` -- Rectangular region coordinates
- `CoordOrigin` -- Enum for coordinate system origin
- `ImageRefMode` -- Enum for image storage modes
- `Size` -- Width/height dimensions

---

## DoclingDocument *(pydantic-model)*

The primary class for representing a complete extracted document.

### Fields

| Field | Type | Default |
|-------|------|---------|
| `body` | `GroupItem` | `GroupItem(name='_root_', self_ref='#/body')` |
| `form_items` | `list[FormItem]` | `[]` |
| `furniture` | `Annotated[GroupItem, Field(deprecated=True)]` | `GroupItem(name='_root_', self_ref='#/furniture', content_layer=FURNITURE)` |
| `groups` | `list[Union[ListGroup, InlineGroup, GroupItem]]` | `[]` |
| `key_value_items` | `list[KeyValueItem]` | `[]` |
| `name` | `str` | *(required)* |
| `origin` | `Optional[DocumentOrigin]` | `None` |
| `pages` | `dict[int, PageItem]` | `{}` |
| `pictures` | `list[PictureItem]` | `[]` |
| `schema_name` | `Literal['DoclingDocument']` | `'DoclingDocument'` |
| `tables` | `list[TableItem]` | `[]` |
| `texts` | `list[Union[TitleItem, SectionHeaderItem, ListItem, CodeItem, FormulaItem, TextItem]]` | `[]` |
| `version` | `Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)]` | `CURRENT_VERSION` |

### Methods

#### add_code

```python
add_code(
    text: str,
    code_language: Optional[CodeLanguageLabel] = None,
    orig: Optional[str] = None,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None
)
```

Add a code block to the document.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Code text content |
| `code_language` | `Optional[CodeLanguageLabel]` | `None` | Programming language label |
| `orig` | `Optional[str]` | `None` | Original text |
| `caption` | `Optional[Union[TextItem, RefItem]]` | `None` | Caption item |
| `prov` | `Optional[ProvenanceItem]` | `None` | Provenance info |
| `parent` | `Optional[NodeItem]` | `None` | Parent node |
| `content_layer` | `Optional[ContentLayer]` | `None` | Content layer |
| `formatting` | `Optional[Formatting]` | `None` | Formatting info |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` | Associated hyperlink |

#### add_comment

```python
add_comment(
    *,
    text: str,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    targets: Optional[list[Union[DocItem, tuple[DocItem, tuple[int, int]]]]] = None
)
```

Adds a comment to the document, assigning it to the given targets.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Comment text |
| `prov` | `Optional[ProvenanceItem]` | `None` | Provenance info |
| `parent` | `Optional[NodeItem]` | `None` | Parent node |
| `targets` | `Optional[list[Union[DocItem, tuple[DocItem, tuple[int, int]]]]]` | `None` | Target items. Each element can be a DocItem or a tuple of (DocItem, (start_inclusive, end_exclusive)) span range. |

#### add_document

```python
add_document(
    doc: DoclingDocument,
    parent: Optional[NodeItem] = None
) -> None
```

Adds the content from the body of a DoclingDocument to this document under a specific parent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doc` | `DoclingDocument` | *(required)* | The document whose content will be added |
| `parent` | `Optional[NodeItem]` | `None` | Parent NodeItem under which new items are added |

#### add_form

```python
add_form(
    graph: GraphData,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `graph` | `GraphData` | *(required)* |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |

#### add_formula

```python
add_formula(
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `text` | `str` | *(required)* |
| `orig` | `Optional[str]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` |

#### add_group

```python
add_group(
    label: Optional[GroupLabel] = None,
    name: Optional[str] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
) -> GroupItem
```

| Parameter | Type | Default |
|-----------|------|---------|
| `label` | `Optional[GroupLabel]` | `None` |
| `name` | `Optional[str]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |

#### add_heading

```python
add_heading(
    text: str,
    orig: Optional[str] = None,
    level: LevelNumber = 1,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `text` | `str` | *(required)* |
| `orig` | `Optional[str]` | `None` |
| `level` | `LevelNumber` | `1` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` |

#### add_inline_group

```python
add_inline_group(
    name: Optional[str] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
) -> InlineGroup
```

#### add_key_values

```python
add_key_values(
    graph: GraphData,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `graph` | `GraphData` | *(required)* |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |

#### add_list_group

```python
add_list_group(
    name: Optional[str] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
) -> ListGroup
```

#### add_list_item

```python
add_list_item(
    text: str,
    enumerated: bool = False,
    marker: Optional[str] = None,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `text` | `str` | *(required)* |
| `enumerated` | `bool` | `False` |
| `marker` | `Optional[str]` | `None` |
| `orig` | `Optional[str]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` |

#### add_node_items

```python
add_node_items(
    node_items: list[NodeItem],
    doc: DoclingDocument,
    parent: Optional[NodeItem] = None
) -> None
```

Adds multiple NodeItems and their children under a parent in this document.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_items` | `list[NodeItem]` | *(required)* | The NodeItems to be added |
| `doc` | `DoclingDocument` | *(required)* | The document to which the NodeItems belong |
| `parent` | `Optional[NodeItem]` | `None` | Parent NodeItem under which new items are added |

#### add_ordered_list

```python
add_ordered_list(
    name: Optional[str] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
) -> GroupItem
```

#### add_page

```python
add_page(
    page_no: int,
    size: Size
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `page_no` | `int` | *(required)* |
| `size` | `Size` | *(required)* |

#### add_picture

```python
add_picture(
    annotations: Optional[list[PictureDataType]] = None,
    image: Optional[ImageRef] = None,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `annotations` | `Optional[list[PictureDataType]]` | `None` |
| `image` | `Optional[ImageRef]` | `None` |
| `caption` | `Optional[Union[TextItem, RefItem]]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |

#### add_table

```python
add_table(
    data: TableData,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    label: DocItemLabel = DocItemLabel.TABLE
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data` | `TableData` | *(required)* |
| `caption` | `Optional[Union[TextItem, RefItem]]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `label` | `DocItemLabel` | `DocItemLabel.TABLE` |

#### add_table_cell

Add a table cell to the table.

#### add_text

```python
add_text(
    label: DocItemLabel,
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    *,
    source: Optional[SourceType] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `label` | `DocItemLabel` | *(required)* |
| `text` | `str` | *(required)* |
| `orig` | `Optional[str]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` |
| `source` | `Optional[SourceType]` | `None` |

#### add_title

```python
add_title(
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `text` | `str` | *(required)* |
| `orig` | `Optional[str]` | `None` |
| `prov` | `Optional[ProvenanceItem]` | `None` |
| `parent` | `Optional[NodeItem]` | `None` |
| `content_layer` | `Optional[ContentLayer]` | `None` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` |

#### add_unordered_list

```python
add_unordered_list(
    name: Optional[str] = None,
    parent: Optional[NodeItem] = None,
    content_layer: Optional[ContentLayer] = None
) -> GroupItem
```

#### append_child_item

```python
append_child_item(
    *,
    child: NodeItem,
    parent: Optional[NodeItem] = None
) -> None
```

Adds an item as a child of the given parent.

#### check_version_is_compatible

```python
check_version_is_compatible(v: str) -> str
```

Check if this document version is compatible with SDK schema version.

#### concatenate

```python
concatenate(*docs) -> DoclingDocument
```

Concatenate multiple documents into a single document. Class method.

#### delete_items

```python
delete_items(*, node_items: list[NodeItem]) -> None
```

Deletes items, given their instances or refs, and any children they have.

#### delete_items_range

```python
delete_items_range(
    *,
    start: NodeItem,
    end: NodeItem,
    start_inclusive: bool = True,
    end_inclusive: bool = True
) -> None
```

Deletes all NodeItems and their children in the range from start to end.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `NodeItem` | *(required)* | Starting NodeItem of the range |
| `end` | `NodeItem` | *(required)* | Ending NodeItem of the range |
| `start_inclusive` | `bool` | `True` | If True, the start NodeItem will also be deleted |
| `end_inclusive` | `bool` | `True` | If True, the end NodeItem will also be deleted |

#### export_to_dict

```python
export_to_dict(
    mode: str = 'json',
    by_alias: bool = True,
    exclude_none: bool = True,
    coord_precision: Optional[int] = None,
    confid_precision: Optional[int] = None
) -> dict[str, Any]
```

Export to dict.

#### export_to_doctags

```python
export_to_doctags(
    delim: str = '',
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_content: bool = True,
    add_page_index: bool = True,
    add_table_cell_location: bool = False,
    add_table_cell_text: bool = True,
    minified: bool = False,
    pages: Optional[set[int]] = None
) -> str
```

Exports the document content to a DocTags format. Operates on a slice of the document's body as defined through `from_element` and `to_element`; defaults to the whole document.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delim` | `str` | `''` | Deprecated |
| `from_element` | `int` | `0` | Start index |
| `to_element` | `int` | `maxsize` | End index |
| `labels` | `Optional[set[DocItemLabel]]` | `None` | Label filter set |
| `xsize` | `int` | `500` | X normalization size |
| `ysize` | `int` | `500` | Y normalization size |
| `add_location` | `bool` | `True` | Include location tokens |
| `add_content` | `bool` | `True` | Include content |
| `add_page_index` | `bool` | `True` | Include page index |
| `add_table_cell_location` | `bool` | `False` | Include table cell locations |
| `add_table_cell_text` | `bool` | `True` | Include table cell text |
| `minified` | `bool` | `False` | Minified output |
| `pages` | `Optional[set[int]]` | `None` | Page filter set |

Returns `str`: The document formatted as a DocTags string.

#### export_to_document_tokens

```python
export_to_document_tokens(*args, **kwargs)
```

Export to DocTags format. (Alias/deprecated wrapper.)

#### export_to_element_tree

```python
export_to_element_tree() -> str
```

#### export_to_html

```python
export_to_html(
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    enable_chart_tables: bool = True,
    image_mode: ImageRefMode = PLACEHOLDER,
    formula_to_mathml: bool = True,
    page_no: Optional[int] = None,
    html_lang: str = 'en',
    html_head: str = 'null',
    included_content_layers: Optional[set[ContentLayer]] = None,
    split_page_view: bool = False,
    include_annotations: bool = True
) -> str
```

Serialize to HTML.

#### export_to_markdown

```python
export_to_markdown(
    delim: str = '\n\n',
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    strict_text: bool = False,
    escape_html: bool = True,
    escape_underscores: bool = True,
    image_placeholder: str = '<!-- image -->',
    enable_chart_tables: bool = True,
    image_mode: ImageRefMode = PLACEHOLDER,
    indent: int = 4,
    text_width: int = -1,
    page_no: Optional[int] = None,
    included_content_layers: Optional[set[ContentLayer]] = None,
    page_break_placeholder: Optional[str] = None,
    include_annotations: bool = True,
    mark_annotations: bool = False,
    compact_tables: bool = False,
    *,
    use_legacy_annotations: Optional[bool] = None,
    allowed_meta_names: Optional[set[str]] = None,
    blocked_meta_names: Optional[set[str]] = None,
    mark_meta: bool = False
) -> str
```

Serialize to Markdown. Operates on a slice of the document's body as defined through `from_element` and `to_element`; defaults to the whole document.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delim` | `str` | `'\n\n'` | Deprecated |
| `from_element` | `int` | `0` | Body slicing start index (inclusive) |
| `to_element` | `int` | `maxsize` | Body slicing stop index (exclusive) |
| `labels` | `Optional[set[DocItemLabel]]` | `None` | Document labels to include. None = system default. |
| `strict_text` | `bool` | `False` | Deprecated |
| `escape_html` | `bool` | `True` | Escape HTML reserved characters in text |
| `escape_underscores` | `bool` | `True` | Escape underscores in text |
| `image_placeholder` | `str` | `'<!-- image -->'` | Placeholder for image positions |
| `enable_chart_tables` | `bool` | `True` | Enable chart table rendering |
| `image_mode` | `ImageRefMode` | `PLACEHOLDER` | Image inclusion mode |
| `indent` | `int` | `4` | Indent in spaces for nested lists |
| `text_width` | `int` | `-1` | Text width (-1 = unlimited) |
| `page_no` | `Optional[int]` | `None` | Filter to single page |
| `included_content_layers` | `Optional[set[ContentLayer]]` | `None` | Content layers to include. None = system default. |
| `page_break_placeholder` | `Optional[str]` | `None` | Page break marker. None = no marker. |
| `include_annotations` | `bool` | `True` | Include annotations (only if item has no meta) |
| `mark_annotations` | `bool` | `False` | Mark annotations in output |
| `compact_tables` | `bool` | `False` | Compact table format without column padding |
| `use_legacy_annotations` | `Optional[bool]` | `None` | Deprecated; legacy annotations only when meta not present |
| `allowed_meta_names` | `Optional[set[str]]` | `None` | Meta names to allow; None = all allowed |
| `blocked_meta_names` | `Optional[set[str]]` | `None` | Meta names to block; takes precedence over allowed |
| `mark_meta` | `bool` | `False` | Mark meta in export |

Returns `str`: The exported Markdown representation.

#### export_to_text

```python
export_to_text(
    delim: str = '\n\n',
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    page_no: Optional[int] = None,
    included_content_layers: Optional[set[ContentLayer]] = None,
    page_break_placeholder: Optional[str] = None
) -> str
```

Export to plain text. Produces clean plain text without any Markdown decoration. Heading markers (`#`), bold/italic markers, and hyperlink syntax are all stripped. List bullets (`-`), ordered list numbers, and table-cell separators (`|`) are preserved.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delim` | `str` | `'\n\n'` | Deprecated |
| `from_element` | `int` | `0` | Body slicing start index (inclusive) |
| `to_element` | `int` | `maxsize` | Body slicing stop index (exclusive) |
| `labels` | `Optional[set[DocItemLabel]]` | `None` | Document labels to include. None = system default. |
| `page_no` | `Optional[int]` | `None` | If set, only content from this page is exported |
| `included_content_layers` | `Optional[set[ContentLayer]]` | `None` | Layers to include. None = system default. |
| `page_break_placeholder` | `Optional[str]` | `None` | String inserted at page boundaries. None = no marker. |

Returns `str`: The exported plain-text representation.

#### export_to_vtt

```python
export_to_vtt(
    included_content_layers: set[ContentLayer] | None = None,
    omit_hours_if_zero: bool = False,
    omit_voice_end: bool = False
) -> str
```

Serializes the document to WebVTT format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `included_content_layers` | `set[ContentLayer] \| None` | `None` | Content layers to serialize. None = `DEFAULT_CONTENT_LAYERS`. |
| `omit_hours_if_zero` | `bool` | `False` | Omit hours when they are 0 in timings |
| `omit_voice_end` | `bool` | `False` | Omit voice end tag for brevity |

#### extract_items_range

```python
extract_items_range(
    start: NodeItem,
    end: NodeItem,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
    delete: bool = False
) -> DoclingDocument
```

Extracts NodeItems and children in the range from start to end as a new DoclingDocument.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `NodeItem` | *(required)* | Starting NodeItem (must be a direct child of body) |
| `end` | `NodeItem` | *(required)* | Ending NodeItem (must be a direct child of body) |
| `start_inclusive` | `bool` | `True` | Include the start NodeItem |
| `end_inclusive` | `bool` | `True` | Include the end NodeItem |
| `delete` | `bool` | `False` | If True, extracted items are deleted from the original |

Returns `DoclingDocument`: A new document containing the extracted items and children.

#### filter

Create a new document based on the provided filter parameters.

#### get_visualization

```python
get_visualization(
    show_label: bool = True,
    show_branch_numbering: bool = False,
    viz_mode: Literal['reading_order', 'key_value'] = 'reading_order',
    show_cell_id: bool = False
) -> dict[Optional[int], Image]
```

Get visualization of the document as images by page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_label` | `bool` | `True` | Show labels on elements |
| `show_branch_numbering` | `bool` | `False` | Show branch numbering (reading order only) |
| `viz_mode` | `Literal['reading_order', 'key_value']` | `'reading_order'` | Which visualizer to use |
| `show_cell_id` | `bool` | `False` | Show cell IDs (key value visualizer only) |

Returns `dict[Optional[int], PILImage.Image]`: Page numbers mapped to PIL images.

#### insert_code

```python
insert_code(
    sibling: NodeItem,
    text: str,
    code_language: Optional[CodeLanguageLabel] = None,
    orig: Optional[str] = None,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> CodeItem
```

Creates a new CodeItem and inserts it into the document relative to a sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sibling` | `NodeItem` | *(required)* | Reference sibling node |
| `text` | `str` | *(required)* | Code text |
| `code_language` | `Optional[CodeLanguageLabel]` | `None` | Language label |
| `orig` | `Optional[str]` | `None` | Original text |
| `caption` | `Optional[Union[TextItem, RefItem]]` | `None` | Caption |
| `prov` | `Optional[ProvenanceItem]` | `None` | Provenance |
| `content_layer` | `Optional[ContentLayer]` | `None` | Content layer |
| `formatting` | `Optional[Formatting]` | `None` | Formatting |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `None` | Hyperlink |
| `after` | `bool` | `True` | Insert after sibling (False = before) |

Returns `CodeItem`.

#### insert_document

```python
insert_document(
    doc: DoclingDocument,
    sibling: NodeItem,
    after: bool = True
) -> None
```

Inserts the content from the body of a DoclingDocument at a specific position.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doc` | `DoclingDocument` | *(required)* | Document whose content will be inserted |
| `sibling` | `NodeItem` | *(required)* | NodeItem after/before which to insert |
| `after` | `bool` | `True` | Insert after sibling (False = before) |

#### insert_form

```python
insert_form(
    sibling: NodeItem,
    graph: GraphData,
    prov: Optional[ProvenanceItem] = None,
    after: bool = True
) -> FormItem
```

Creates a new FormItem and inserts it into the document.

#### insert_formula

```python
insert_formula(
    sibling: NodeItem,
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> FormulaItem
```

Creates a new FormulaItem and inserts it into the document.

#### insert_group

```python
insert_group(
    sibling: NodeItem,
    label: Optional[GroupLabel] = None,
    name: Optional[str] = None,
    content_layer: Optional[ContentLayer] = None,
    after: bool = True
) -> GroupItem
```

Creates a new GroupItem and inserts it into the document.

#### insert_heading

```python
insert_heading(
    sibling: NodeItem,
    text: str,
    orig: Optional[str] = None,
    level: LevelNumber = 1,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> SectionHeaderItem
```

Creates a new SectionHeaderItem and inserts it into the document.

#### insert_inline_group

```python
insert_inline_group(
    sibling: NodeItem,
    name: Optional[str] = None,
    content_layer: Optional[ContentLayer] = None,
    after: bool = True
) -> InlineGroup
```

Creates a new InlineGroup and inserts it into the document.

#### insert_item_after_sibling

Inserts an item, given its node_item instance, after another as a sibling.

#### insert_item_before_sibling

Inserts an item, given its node_item instance, before another as a sibling.

#### insert_key_values

```python
insert_key_values(
    sibling: NodeItem,
    graph: GraphData,
    prov: Optional[ProvenanceItem] = None,
    after: bool = True
) -> KeyValueItem
```

Creates a new KeyValueItem and inserts it into the document.

#### insert_list_group

```python
insert_list_group(
    sibling: NodeItem,
    name: Optional[str] = None,
    content_layer: Optional[ContentLayer] = None,
    after: bool = True
) -> ListGroup
```

Creates a new ListGroup and inserts it into the document.

#### insert_list_item

```python
insert_list_item(
    sibling: NodeItem,
    text: str,
    enumerated: bool = False,
    marker: Optional[str] = None,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> ListItem
```

Creates a new ListItem and inserts it into the document.

#### insert_node_items

```python
insert_node_items(
    sibling: NodeItem,
    node_items: list[NodeItem],
    doc: DoclingDocument,
    after: bool = True
) -> None
```

Insert multiple NodeItems and their children at a specific position.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sibling` | `NodeItem` | *(required)* | NodeItem after/before which to insert |
| `node_items` | `list[NodeItem]` | *(required)* | The NodeItems to be inserted |
| `doc` | `DoclingDocument` | *(required)* | The document to which the NodeItems belong |
| `after` | `bool` | `True` | Insert after sibling (False = before) |

#### insert_picture

```python
insert_picture(
    sibling: NodeItem,
    annotations: Optional[list[PictureDataType]] = None,
    image: Optional[ImageRef] = None,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    after: bool = True
) -> PictureItem
```

Creates a new PictureItem and inserts it into the document.

#### insert_table

```python
insert_table(
    sibling: NodeItem,
    data: TableData,
    caption: Optional[Union[TextItem, RefItem]] = None,
    prov: Optional[ProvenanceItem] = None,
    label: DocItemLabel = DocItemLabel.TABLE,
    content_layer: Optional[ContentLayer] = None,
    annotations: Optional[list[TableAnnotationType]] = None,
    after: bool = True
) -> TableItem
```

Creates a new TableItem and inserts it into the document.

#### insert_text

```python
insert_text(
    sibling: NodeItem,
    label: DocItemLabel,
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> TextItem
```

Creates a new TextItem and inserts it into the document.

#### insert_title

```python
insert_title(
    sibling: NodeItem,
    text: str,
    orig: Optional[str] = None,
    prov: Optional[ProvenanceItem] = None,
    content_layer: Optional[ContentLayer] = None,
    formatting: Optional[Formatting] = None,
    hyperlink: Optional[Union[AnyUrl, Path]] = None,
    after: bool = True
) -> TitleItem
```

Creates a new TitleItem and inserts it into the document.

#### iterate_items

```python
iterate_items(
    root: Optional[NodeItem] = None,
    with_groups: bool = False,
    traverse_pictures: bool = False,
    page_no: Optional[int] = None,
    included_content_layers: Optional[set[ContentLayer]] = None,
    _level: int = 0
) -> Iterable[tuple[NodeItem, int]]
```

Iterate document elements with their nesting level.

#### load_from_doctags

```python
load_from_doctags(
    doctag_document: DocTagsDocument,
    document_name: str = 'Document'
) -> DoclingDocument
```

Load Docling document from lists of DocTags and Images. Class method.

#### load_from_json

```python
load_from_json(filename: Union[str, Path]) -> DoclingDocument
```

Load a DoclingDocument from a `.json` file. Class method.

#### load_from_yaml

```python
load_from_yaml(filename: Union[str, Path]) -> DoclingDocument
```

Load a DoclingDocument from a YAML file. Class method.

#### num_pages

```python
num_pages() -> int
```

#### print_element_tree

```python
print_element_tree()
```

#### replace_item

Replace item with new item.

#### save_as_doctags

```python
save_as_doctags(
    filename: Union[str, Path],
    delim: str = '',
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_content: bool = True,
    add_page_index: bool = True,
    add_table_cell_location: bool = False,
    add_table_cell_text: bool = True,
    minified: bool = False
)
```

Save the document content to DocTags format.

#### save_as_document_tokens

```python
save_as_document_tokens(*args, **kwargs)
```

Save the document content to a DocumentToken format.

#### save_as_html

```python
save_as_html(
    filename: Union[str, Path],
    artifacts_dir: Optional[Path] = None,
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    image_mode: ImageRefMode = PLACEHOLDER,
    formula_to_mathml: bool = True,
    page_no: Optional[int] = None,
    html_lang: str = 'en',
    html_head: str = 'null',
    included_content_layers: Optional[set[ContentLayer]] = None,
    split_page_view: bool = False,
    include_annotations: bool = True
)
```

Save to HTML.

#### save_as_json

```python
save_as_json(
    filename: Union[str, Path],
    artifacts_dir: Optional[Path] = None,
    image_mode: ImageRefMode = EMBEDDED,
    indent: int = 2,
    coord_precision: Optional[int] = None,
    confid_precision: Optional[int] = None
)
```

Save as JSON.

#### save_as_markdown

```python
save_as_markdown(
    filename: Union[str, Path],
    artifacts_dir: Optional[Path] = None,
    delim: str = '\n\n',
    from_element: int = 0,
    to_element: int = maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    strict_text: bool = False,
    escape_html: bool = True,
    escaping_underscores: bool = True,
    image_placeholder: str = '<!-- image -->',
    image_mode: ImageRefMode = PLACEHOLDER,
    indent: int = 4,
    text_width: int = -1,
    page_no: Optional[int] = None,
    included_content_layers: Optional[set[ContentLayer]] = None,
    page_break_placeholder: Optional[str] = None,
    include_annotations: bool = True,
    compact_tables: bool = False,
    *,
    mark_meta: bool = False,
    use_legacy_annotations: Optional[bool] = None
)
```

Save to Markdown.

#### save_as_vtt

```python
save_as_vtt(
    filename: str | Path,
    included_content_layers: set[ContentLayer] | None = None,
    omit_hours_if_zero: bool = False,
    omit_voice_end: bool = True
) -> None
```

Saves the document to a file in WebVTT format.

#### save_as_yaml

```python
save_as_yaml(
    filename: Union[str, Path],
    artifacts_dir: Optional[Path] = None,
    image_mode: ImageRefMode = EMBEDDED,
    default_flow_style: bool = False,
    coord_precision: Optional[int] = None,
    confid_precision: Optional[int] = None
)
```

Save as YAML.

#### transform_to_content_layer

```python
transform_to_content_layer(data: Any) -> Any
```

#### validate_document

```python
validate_document() -> Self
```

#### validate_misplaced_list_items

```python
validate_misplaced_list_items() -> Self
```

#### validate_tree

Validate the document tree structure.

---

## DocumentOrigin *(pydantic-model)*

Bases: `BaseModel`

FileSource.

| Field | Type | Default |
|-------|------|---------|
| `binary_hash` | `Uint64` | *(required)* |
| `filename` | `str` | *(required)* |
| `mimetype` | `str` | *(required)* |
| `uri` | `Optional[AnyUrl]` | `None` |

### Methods

#### parse_hex_string

```python
parse_hex_string(value)
```

#### validate_mimetype

```python
validate_mimetype(v)
```

---

## DocItem *(pydantic-model)*

Bases: `NodeItem`

Base type for any element that carries content, can be a leaf node.

| Field | Type | Default |
|-------|------|---------|
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `label` | `DocItemLabel` | *(required)* |
| `meta` | `Optional[BaseMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |

### Methods

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

Get the annotations of this DocItem.

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

Returns the image of this DocItem. Returns None if this DocItem has no valid provenance or if a valid image of the page containing this DocItem is not available in doc.

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

Get the location string for the BaseCell.

#### get_ref

```python
get_ref() -> RefItem
```

---

## DocItemLabel *(enum)*

Bases: `str`, `Enum`

| Value | String |
|-------|--------|
| `CAPTION` | `'caption'` |
| `CHART` | `'chart'` |
| `CHECKBOX_SELECTED` | `'checkbox_selected'` |
| `CHECKBOX_UNSELECTED` | `'checkbox_unselected'` |
| `CODE` | `'code'` |
| `DOCUMENT_INDEX` | `'document_index'` |
| `EMPTY_VALUE` | `'empty_value'` |
| `FOOTNOTE` | `'footnote'` |
| `FORM` | `'form'` |
| `FORMULA` | `'formula'` |
| `GRADING_SCALE` | `'grading_scale'` |
| `HANDWRITTEN_TEXT` | `'handwritten_text'` |
| `KEY_VALUE_REGION` | `'key_value_region'` |
| `LIST_ITEM` | `'list_item'` |
| `PAGE_FOOTER` | `'page_footer'` |
| `PAGE_HEADER` | `'page_header'` |
| `PARAGRAPH` | `'paragraph'` |
| `PICTURE` | `'picture'` |
| `REFERENCE` | `'reference'` |
| `SECTION_HEADER` | `'section_header'` |
| `TABLE` | `'table'` |
| `TEXT` | `'text'` |
| `TITLE` | `'title'` |

### Methods

#### get_color

```python
get_color(label: DocItemLabel) -> tuple[int, int, int]
```

Return the RGB color associated with a given label.

---

## ProvenanceItem *(pydantic-model)*

Bases: `BaseModel`

Provenance information for elements extracted from a textual document. A ProvenanceItem object acts as a lightweight pointer back into the original document for an extracted element. It applies to documents with an explicit or implicit layout, such as PDF, HTML, docx, or pptx.

| Field | Type | Default |
|-------|------|---------|
| `bbox` | `Annotated[BoundingBox, Field(description='Bounding box')]` | *(required)* |
| `charspan` | `Annotated[tuple[int, int], Field(description='Character span (0-indexed)')]` | *(required)* |
| `page_no` | `Annotated[int, Field(description='Page number')]` | *(required)* |

---

## GroupItem *(pydantic-model)*

Bases: `NodeItem`

| Field | Type | Default |
|-------|------|---------|
| `children` | `list[RefItem]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `label` | `GroupLabel` | `UNSPECIFIED` |
| `meta` | `Optional[BaseMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `name` | `str` | `'group'` |
| `parent` | `Optional[RefItem]` | `None` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |

### Methods

#### get_ref

```python
get_ref() -> RefItem
```

---

## GroupLabel *(enum)*

Bases: `str`, `Enum`

| Value | String |
|-------|--------|
| `CHAPTER` | `'chapter'` |
| `COMMENT_SECTION` | `'comment_section'` |
| `FORM_AREA` | `'form_area'` |
| `INLINE` | `'inline'` |
| `KEY_VALUE_AREA` | `'key_value_area'` |
| `LIST` | `'list'` |
| `ORDERED_LIST` | `'ordered_list'` |
| `PICTURE_AREA` | `'picture_area'` |
| `SECTION` | `'section'` |
| `SHEET` | `'sheet'` |
| `SLIDE` | `'slide'` |
| `UNSPECIFIED` | `'unspecified'` |

---

## NodeItem *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `children` | `list[RefItem]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `meta` | `Optional[BaseMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |

### Methods

#### get_ref

```python
get_ref() -> RefItem
```

---

## PageItem *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `image` | `Optional[ImageRef]` | `None` |
| `page_no` | `int` | *(required)* |
| `size` | `Size` | *(required)* |

---

## FloatingItem *(pydantic-model)*

Bases: `DocItem`

*Inherits all fields from DocItem.*

| Field | Type | Default |
|-------|------|---------|
| `captions` | `list[RefItem]` | `[]` |
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `footnotes` | `list[RefItem]` | `[]` |
| `image` | `Optional[ImageRef]` | `None` |
| `label` | `DocItemLabel` | *(required)* |
| `meta` | `Optional[FloatingMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `references` | `list[RefItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |

### Methods

#### caption_text

`str`: Combined caption text.

### Methods

#### caption_text

```python
caption_text(doc: DoclingDocument) -> str
```

Computes the caption as a single text.

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

Get the annotations of this DocItem.

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

Returns the image corresponding to this FloatingItem. Returns the PIL image from `self.image` if available. Otherwise, uses `DocItem.get_image`. Returns None if no valid provenance or no valid page image in doc.

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

Get the location string for the BaseCell.

#### get_ref

```python
get_ref() -> RefItem
```

---

## TextItem *(pydantic-model)*

Bases: `DocItem`

| Field | Type | Default |
|-------|------|---------|
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `Field(union_mode='left_to_right', default=None)` |
| `label` | `Literal[CAPTION, CHECKBOX_SELECTED, CHECKBOX_UNSELECTED, FOOTNOTE, PAGE_FOOTER, PAGE_HEADER, PARAGRAPH, REFERENCE, TEXT, EMPTY_VALUE]` | *(required)* |
| `meta` | `Optional[BaseMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `orig` | `str` | *(required)* |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |
| `text` | `str` | *(required)* |

### Methods

#### export_to_doctags

```python
export_to_doctags(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_content: bool = True
)
```

Export text element to document tokens format.

| Parameter | Type | Default |
|-----------|------|---------|
| `doc` | `DoclingDocument` | *(required)* |
| `new_line` | `str` | `''` (Deprecated) |
| `xsize` | `int` | `500` |
| `ysize` | `int` | `500` |
| `add_location` | `bool` | `True` |
| `add_content` | `bool` | `True` |

#### export_to_document_tokens

```python
export_to_document_tokens(*args, **kwargs)
```

Export to DocTags format.

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

#### get_ref

```python
get_ref() -> RefItem
```

---

## TableItem *(pydantic-model)*

Bases: `FloatingItem`

*Inherits all fields from FloatingItem.*

| Field | Type | Default |
|-------|------|---------|
| `annotations` | `Annotated[list[TableAnnotationType], deprecated('Field annotations is deprecated; use meta instead.')]` | `[]` |
| `captions` | `list[RefItem]` | `[]` |
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `data` | `TableData` | *(required)* |
| `footnotes` | `list[RefItem]` | `[]` |
| `image` | `Optional[ImageRef]` | `None` |
| `label` | `Literal[DOCUMENT_INDEX, TABLE]` | `TABLE` |
| `meta` | `Optional[FloatingMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `references` | `list[RefItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |

### Methods

#### add_annotation

```python
add_annotation(annotation: TableAnnotationType) -> None
```

Add an annotation to the table.

#### caption_text

```python
caption_text(doc: DoclingDocument) -> str
```

Computes the caption as a single text.

#### export_to_dataframe

```python
export_to_dataframe(doc: Optional[DoclingDocument] = None) -> DataFrame
```

Export the table as a Pandas DataFrame.

#### export_to_doctags

```python
export_to_doctags(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_cell_location: bool = True,
    add_cell_text: bool = True,
    add_caption: bool = True
)
```

Export table to document tokens format.

| Parameter | Type | Default |
|-----------|------|---------|
| `doc` | `DoclingDocument` | *(required)* |
| `new_line` | `str` | `''` (Deprecated) |
| `xsize` | `int` | `500` |
| `ysize` | `int` | `500` |
| `add_location` | `bool` | `True` |
| `add_cell_location` | `bool` | `True` |
| `add_cell_text` | `bool` | `True` |
| `add_caption` | `bool` | `True` |

#### export_to_document_tokens

```python
export_to_document_tokens(*args, **kwargs)
```

Export to DocTags format.

#### export_to_html

```python
export_to_html(doc: Optional[DoclingDocument] = None, add_caption: bool = True) -> str
```

Export the table as HTML.

#### export_to_markdown

```python
export_to_markdown(doc: Optional[DoclingDocument] = None) -> str
```

Export the table as markdown.

#### export_to_otsl

```python
export_to_otsl(
    doc: DoclingDocument,
    add_cell_location: bool = True,
    add_cell_text: bool = True,
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False,
    **kwargs: Any
) -> str
```

Export the table as OTSL.

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

Get the annotations of this TableItem.

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

#### get_ref

```python
get_ref() -> RefItem
```

---

## TableCell *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `bbox` | `Optional[BoundingBox]` | `None` |
| `col_span` | `int` | `1` |
| `column_header` | `bool` | `False` |
| `end_col_offset_idx` | `int` | *(required)* |
| `end_row_offset_idx` | `int` | *(required)* |
| `fillable` | `bool` | `False` |
| `row_header` | `bool` | `False` |
| `row_section` | `bool` | `False` |
| `row_span` | `int` | `1` |
| `start_col_offset_idx` | `int` | *(required)* |
| `start_row_offset_idx` | `int` | *(required)* |
| `text` | `str` | *(required)* |

### Methods

#### from_dict_format

```python
from_dict_format(data: Any) -> Any
```

---

## TableData *(pydantic-model)*

Bases: `BaseModel`

BaseTableData.

| Field | Type | Default |
|-------|------|---------|
| `grid` | `list[list[TableCell]]` | *(required)* |
| `num_cols` | `int` | `0` |
| `num_rows` | `int` | `0` |
| `table_cells` | `list[AnyTableCell]` | `[]` |

### Methods

#### add_row

```python
add_row(row: list[str]) -> None
```

Add a new row to the table from a list of strings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | `list[str]` | A list of strings representing the content of the new row |

#### add_rows

```python
add_rows(rows: list[list[str]]) -> None
```

Add multiple new rows to the table from a list of lists of strings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `list[list[str]]` | A list of lists, where each inner list represents the content of a new row |

#### from_regions

```python
from_regions(
    table_bbox: BoundingBox,
    rows: list[BoundingBox],
    cols: list[BoundingBox],
    merges: list[BoundingBox],
    row_headers: list[BoundingBox] = [],
    col_headers: list[BoundingBox] = [],
    row_sections: list[BoundingBox] = []
) -> Self
```

Converts regions: rows, columns, merged cells into table_data structure. Adds semantics for regions of row_headers, col_headers, row_section. Class method.

#### get_column_bounding_boxes

```python
get_column_bounding_boxes(*, minimal: bool = True) -> dict[int, BoundingBox]
```

Get the bounding box for each column in the table. If `minimal=True` (default), returns the minimal bounding box for each column based on its cells. If `False`, all columns will have uniform vertical extent spanning the full table height. Only columns with cells that have bounding boxes are included.

#### get_row_bounding_boxes

```python
get_row_bounding_boxes(*, minimal: bool = True) -> dict[int, BoundingBox]
```

Get the bounding box for each row in the table. If `minimal=True` (default), returns the minimal bounding box for each row based on its cells. If `False`, all rows will have uniform horizontal extent spanning the full table width. Only rows with cells that have bounding boxes are included.

#### insert_row

```python
insert_row(row_index: int, row: list[str], after: bool = False) -> None
```

Insert a new row from a list of strings before/after a specific index in the table.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row_index` | `int` | *(required)* | The index at which to insert the new row (starting from 0) |
| `row` | `list[str]` | *(required)* | A list of strings representing the content of the new row |
| `after` | `bool` | `False` | If True, insert after the specified index, otherwise before |

#### insert_rows

```python
insert_rows(row_index: int, rows: list[list[str]], after: bool = False) -> None
```

Insert multiple new rows from a list of lists of strings before/after a specific index in the table.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row_index` | `int` | *(required)* | The index at which to insert the new rows (starting from 0) |
| `rows` | `list[list[str]]` | *(required)* | A list of lists, where each inner list represents the content of a new row |
| `after` | `bool` | `False` | If True, insert after the specified index, otherwise before |

#### pop_row

```python
pop_row(doc: Optional[DoclingDocument] = None) -> list[TableCell]
```

Remove and return the last row from the table.

#### remove_row

```python
remove_row(row_index: int, doc: Optional[DoclingDocument] = None) -> list[TableCell]
```

Remove a row from the table by its index.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row_index` | `int` | The index of the row to remove (starting from 0) |

#### remove_rows

```python
remove_rows(indices: list[int], doc: Optional[DoclingDocument] = None) -> list[list[TableCell]]
```

Remove rows from the table by their indices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `indices` | `list[int]` | A list of indices of the rows to remove (starting from 0) |

---

## TableCellLabel *(enum)*

Bases: `str`, `Enum`

| Value | String |
|-------|--------|
| `BODY` | `'body'` |
| `COLUMN_HEADER` | `'col_header'` |
| `ROW_HEADER` | `'row_header'` |
| `ROW_SECTION` | `'row_section'` |

### Methods

#### get_color

```python
get_color(label: TableCellLabel) -> tuple[int, int, int]
```

Return the RGB color associated with a given label.

---

## KeyValueItem *(pydantic-model)*

Bases: `FloatingItem`

*Inherits all fields from FloatingItem.*

| Field | Type | Default |
|-------|------|---------|
| `captions` | `list[RefItem]` | `[]` |
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `footnotes` | `list[RefItem]` | `[]` |
| `graph` | `GraphData` | *(required)* |
| `image` | `Optional[ImageRef]` | `None` |
| `label` | `Literal[KEY_VALUE_REGION]` | `KEY_VALUE_REGION` |
| `meta` | `Optional[FloatingMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `references` | `list[RefItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |

### Methods

#### caption_text

```python
caption_text(doc: DoclingDocument) -> str
```

Computes the caption as a single text.

#### export_to_document_tokens

```python
export_to_document_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_content: bool = True
)
```

Export key value item to document tokens format.

| Parameter | Type | Default |
|-----------|------|---------|
| `doc` | `DoclingDocument` | *(required)* |
| `new_line` | `str` | `''` (Deprecated) |
| `xsize` | `int` | `500` |
| `ysize` | `int` | `500` |
| `add_location` | `bool` | `True` |
| `add_content` | `bool` | `True` |

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

#### get_ref

```python
get_ref() -> RefItem
```

---

## SectionHeaderItem *(pydantic-model)*

Bases: `TextItem`

*Inherits all fields from TextItem.*

| Field | Type | Default |
|-------|------|---------|
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `formatting` | `Optional[Formatting]` | `None` |
| `hyperlink` | `Optional[Union[AnyUrl, Path]]` | `Field(union_mode='left_to_right', default=None)` |
| `label` | `Literal[SECTION_HEADER]` | `SECTION_HEADER` |
| `level` | `LevelNumber` | `1` |
| `meta` | `Optional[BaseMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `orig` | `str` | *(required)* |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |
| `text` | `str` | *(required)* |

### Methods

#### export_to_doctags

```python
export_to_doctags(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_content: bool = True
)
```

Export text element to document tokens format.

#### export_to_document_tokens

```python
export_to_document_tokens(*args, **kwargs)
```

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

#### get_ref

```python
get_ref() -> RefItem
```

---

## PictureItem *(pydantic-model)*

Bases: `FloatingItem`

*Inherits all fields from FloatingItem.*

| Field | Type | Default |
|-------|------|---------|
| `annotations` | `Annotated[list[PictureDataType], deprecated('Field annotations is deprecated; use meta instead.')]` | `[]` |
| `captions` | `list[RefItem]` | `[]` |
| `children` | `list[RefItem]` | `[]` |
| `comments` | `list[FineRef]` | `[]` |
| `content_layer` | `ContentLayer` | `BODY` |
| `footnotes` | `list[RefItem]` | `[]` |
| `image` | `Optional[ImageRef]` | `None` |
| `label` | `Literal[PICTURE, CHART]` | `PICTURE` |
| `meta` | `Optional[PictureMeta]` | `None` |
| `model_config` | | `ConfigDict(extra='forbid')` |
| `parent` | `Optional[RefItem]` | `None` |
| `prov` | `list[ProvenanceItem]` | `[]` |
| `references` | `list[RefItem]` | `[]` |
| `self_ref` | `str` | `Field(pattern=_JSON_POINTER_REGEX)` |
| `source` | `Annotated[list[SourceType], Field(description='The provenance of this document item. Currently, it is only used for media track provenance.')]` | `[]` |

### Methods

#### caption_text

```python
caption_text(doc: DoclingDocument) -> str
```

Computes the caption as a single text.

#### export_to_doctags

```python
export_to_doctags(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    add_location: bool = True,
    add_caption: bool = True,
    add_content: bool = True
)
```

Export picture to document tokens format.

| Parameter | Type | Default |
|-----------|------|---------|
| `doc` | `DoclingDocument` | *(required)* |
| `new_line` | `str` | `''` (Deprecated) |
| `xsize` | `int` | `500` |
| `ysize` | `int` | `500` |
| `add_location` | `bool` | `True` |
| `add_caption` | `bool` | `True` |
| `add_content` | `bool` | `True` |

#### export_to_document_tokens

```python
export_to_document_tokens(*args, **kwargs)
```

#### export_to_html

```python
export_to_html(
    doc: DoclingDocument,
    add_caption: bool = True,
    image_mode: ImageRefMode = PLACEHOLDER
) -> str
```

Export picture to HTML format.

#### export_to_markdown

```python
export_to_markdown(
    doc: DoclingDocument,
    add_caption: bool = True,
    image_mode: ImageRefMode = EMBEDDED,
    image_placeholder: str = '<!-- image -->'
) -> str
```

Export picture to Markdown format.

#### get_annotations

```python
get_annotations() -> Sequence[BaseAnnotation]
```

Get the annotations of this PictureItem.

#### get_image

```python
get_image(doc: DoclingDocument, prov_index: int = 0) -> Optional[Image]
```

#### get_location_tokens

```python
get_location_tokens(
    doc: DoclingDocument,
    new_line: str = '',
    xsize: int = 500,
    ysize: int = 500,
    self_closing: bool = False
) -> str
```

#### get_ref

```python
get_ref() -> RefItem
```

---

## ImageRef *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `dpi` | `int` | *(required)* |
| `mimetype` | `str` | *(required)* |
| `pil_image` | `Optional[Image]` | *(property)* Return the PIL Image. |
| `size` | `Size` | *(required)* |
| `uri` | `Union[AnyUrl, Path]` | `Field(union_mode='left_to_right')` |

### Methods

#### from_pil

```python
from_pil(image: Image, dpi: int) -> Self
```

Construct ImageRef from a PIL Image. Class method.

#### validate_mimetype

```python
validate_mimetype(v)
```

---

## PictureClassificationClass *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `class_name` | `str` | *(required)* |
| `confidence` | `float` | *(required)* |

---

## PictureClassificationData *(pydantic-model)*

Bases: `BaseAnnotation`

| Field | Type | Default |
|-------|------|---------|
| `kind` | `Literal['classification']` | `'classification'` |
| `predicted_classes` | `list[PictureClassificationClass]` | *(required)* |
| `provenance` | `str` | *(required)* |

---

## RefItem *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `cref` | `str` | `Field(alias='$ref', pattern=_JSON_POINTER_REGEX)` |
| `model_config` | | `ConfigDict(populate_by_name=True)` |

### Methods

#### get_ref

```python
get_ref()
```

#### resolve

```python
resolve(doc: DoclingDocument)
```

Resolve the path in the document.

---

## BoundingBox *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `b` | `float` | *(required)* |
| `coord_origin` | `CoordOrigin` | `TOPLEFT` |
| `l` | `float` | *(required)* |
| `r` | `float` | *(required)* |
| `t` | `float` | *(required)* |

### Computed Properties

| Property | Type |
|----------|------|
| `height` | `float` |
| `width` | `float` |

### Methods

#### area

```python
area() -> float
```

#### as_tuple

```python
as_tuple() -> tuple[float, float, float, float]
```

#### enclosing_bbox

```python
enclosing_bbox(boxes: list[BoundingBox]) -> BoundingBox
```

Create a bounding box that covers all of the given boxes.

#### expand_by_scale

```python
expand_by_scale(x_scale: float, y_scale: float) -> BoundingBox
```

#### from_tuple

```python
from_tuple(coord: tuple[float, ...], origin: CoordOrigin)
```

| Parameter | Type |
|-----------|------|
| `coord` | `tuple[float, ...]` |
| `origin` | `CoordOrigin` |

#### get_intersection_bbox

```python
get_intersection_bbox(other: BoundingBox) -> Optional[BoundingBox]
```

Return the intersection bounding box with another bounding box or None when disjoint.

#### intersection_area_with

```python
intersection_area_with(other: BoundingBox) -> float
```

Calculate the intersection area with another bounding box.

#### intersection_over_self

```python
intersection_over_self(other: BoundingBox, eps: float = 1e-06) -> float
```

#### intersection_over_union

```python
intersection_over_union(other: BoundingBox, eps: float = 1e-06) -> float
```

#### is_above

```python
is_above(other: BoundingBox) -> bool
```

#### is_horizontally_connected

```python
is_horizontally_connected(elem_i: BoundingBox, elem_j: BoundingBox) -> bool
```

#### is_left_of

```python
is_left_of(other: BoundingBox) -> bool
```

#### is_strictly_above

```python
is_strictly_above(other: BoundingBox, eps: float = 0.001) -> bool
```

#### is_strictly_left_of

```python
is_strictly_left_of(other: BoundingBox, eps: float = 0.001) -> bool
```

#### normalized

```python
normalized(page_size: Size)
```

#### overlaps

```python
overlaps(other: BoundingBox) -> bool
```

#### overlaps_horizontally

```python
overlaps_horizontally(other: BoundingBox) -> bool
```

Check if two bounding boxes overlap horizontally.

#### overlaps_vertically

```python
overlaps_vertically(other: BoundingBox) -> bool
```

Check if two bounding boxes overlap vertically.

#### overlaps_vertically_with_iou

```python
overlaps_vertically_with_iou(other: BoundingBox, iou: float) -> bool
```

#### resize_by_scale

```python
resize_by_scale(x_scale: float, y_scale: float)
```

#### scale_to_size

```python
scale_to_size(old_size: Size, new_size: Size)
```

#### scaled

```python
scaled(scale: float)
```

#### to_bottom_left_origin

```python
to_bottom_left_origin(page_height: float) -> BoundingBox
```

#### to_top_left_origin

```python
to_top_left_origin(page_height: float) -> BoundingBox
```

#### union_area_with

```python
union_area_with(other: BoundingBox) -> float
```

Calculates the union area with another bounding box.

#### x_overlap_with

```python
x_overlap_with(other: BoundingBox) -> float
```

Calculates the horizontal overlap with another bounding box.

#### x_union_with

```python
x_union_with(other: BoundingBox) -> float
```

Calculates the horizontal union dimension with another bounding box.

#### y_overlap_with

```python
y_overlap_with(other: BoundingBox) -> float
```

Calculates the vertical overlap with another bounding box, respecting coordinate origin.

#### y_union_with

```python
y_union_with(other: BoundingBox) -> float
```

Calculates the vertical union dimension with another bounding box, respecting coordinate origin.

---

## CoordOrigin *(enum)*

Bases: `str`, `Enum`

| Value | String |
|-------|--------|
| `BOTTOMLEFT` | `'BOTTOMLEFT'` |
| `TOPLEFT` | `'TOPLEFT'` |

---

## ImageRefMode *(enum)*

Bases: `str`, `Enum`

| Value | String |
|-------|--------|
| `EMBEDDED` | `'embedded'` |
| `PLACEHOLDER` | `'placeholder'` |
| `REFERENCED` | `'referenced'` |

---

## Size *(pydantic-model)*

Bases: `BaseModel`

| Field | Type | Default |
|-------|------|---------|
| `height` | `float` | `0.0` |
| `width` | `float` | `0.0` |

### Methods

#### as_tuple

```python
as_tuple()
```
