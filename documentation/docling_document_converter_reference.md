# Document Converter

This is an automatic generated API reference of the main components of Docling.

## document_converter

### Classes

- `DocumentConverter` -- Convert documents of various input formats to Docling documents.
- `ConversionResult` -- Conversion result wrapper.
- `ConversionStatus` -- Conversion status enum.
- `FormatOption` -- Base format option.
- `InputFormat` -- A document format supported by document backend parsers.
- `PdfFormatOption` -- PDF format option.
- `ImageFormatOption` -- Image format option.
- `StandardPdfPipeline` -- High-performance PDF pipeline with multi-threaded stages.
- `WordFormatOption` -- Word format option.
- `PowerpointFormatOption` -- Powerpoint format option.
- `MarkdownFormatOption` -- Markdown format option.
- `AsciiDocFormatOption` -- AsciiDoc format option.
- `HTMLFormatOption` -- HTML format option.
- `SimplePipeline` -- SimpleModelPipeline.

---

## DocumentConverter

```python
DocumentConverter(
    allowed_formats: Optional[list[InputFormat]] = None,
    format_options: Optional[dict[InputFormat, FormatOption]] = None
)
```

Convert documents of various input formats to Docling documents.

`DocumentConverter` is the main entry point for converting documents in Docling.
It handles various input formats (PDF, DOCX, PPTX, images, HTML, Markdown, etc.)
and provides both single-document and batch conversion capabilities.

The conversion methods return a `ConversionResult` instance for each document,
which wraps a `DoclingDocument` object if the conversion was successful, along
with metadata about the conversion process.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `allowed_formats` | `Optional[list[InputFormat]]` | `None` | List of allowed input formats. By default, any format supported by Docling is allowed. |
| `format_options` | `Optional[dict[InputFormat, FormatOption]]` | `None` | Dictionary of format-specific options. |

**Examples:**

Create a converter with default settings (all formats allowed):

```python
>>> converter = DocumentConverter()
```

Allow only PDF and DOCX formats:

```python
>>> from docling.datamodel.base_models import InputFormat
>>> converter = DocumentConverter(
...     allowed_formats=[InputFormat.PDF, InputFormat.DOCX]
... )
```

Customize pipeline options for PDF:

```python
>>> from docling.datamodel.pipeline_options import PdfPipelineOptions
>>> converter = DocumentConverter(
...     format_options={
...         InputFormat.PDF: PdfFormatOption(
...             pipeline_options=PdfPipelineOptions()
...         ),
...     }
... )
```

### Attributes

#### allowed_formats *(instance-attribute)*

```python
allowed_formats: list[InputFormat]
```

#### format_to_options *(instance-attribute)*

```python
format_to_options: dict[InputFormat, FormatOption]
```

#### initialized_pipelines *(instance-attribute)*

```python
initialized_pipelines: dict[tuple[Type[BasePipeline], str], BasePipeline]
```

### Methods

#### convert

```python
convert(
    source: Union[Path, str, DocumentStream],
    headers: Optional[dict[str, str]] = None,
    raises_on_error: bool = True,
    max_num_pages: int = maxsize,
    max_file_size: int = maxsize,
    page_range: PageRange = DEFAULT_PAGE_RANGE
) -> ConversionResult
```

Convert one document fetched from a file path, URL, or DocumentStream.

Note: If the document content is given as a string (Markdown or HTML content), use the `convert_string` method.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `Union[Path, str, DocumentStream]` | *(required)* | Source of input document given as file path, URL, or DocumentStream. |
| `headers` | `Optional[dict[str, str]]` | `None` | Optional headers given as a dictionary of string key-value pairs, in case of URL input source. |
| `raises_on_error` | `bool` | `True` | Whether to raise an error on the first conversion failure. If False, errors are captured in the ConversionResult objects. |
| `max_num_pages` | `int` | `maxsize` | Maximum number of pages accepted per document. Documents exceeding this number will not be converted. |
| `max_file_size` | `int` | `maxsize` | Maximum file size to convert. |
| `page_range` | `PageRange` | `DEFAULT_PAGE_RANGE` | Range of pages to convert. |

**Returns:**

| Type | Description |
|------|-------------|
| `ConversionResult` | The conversion result, which contains a `DoclingDocument` in the `document` attribute, and metadata about the conversion process. |

**Raises:**

| Exception | Description |
|-----------|-------------|
| `ConversionError` | An error occurred during conversion. |

**Examples:**

Convert a local PDF file:

```python
>>> from pathlib import Path
>>> converter = DocumentConverter()
>>> result = converter.convert("path/to/document.pdf")
>>> print(result.document.export_to_markdown())
```

Convert a document from a URL:

```python
>>> result = converter.convert("https://example.com/paper.pdf")
```

Convert from an in-memory stream:

```python
>>> from io import BytesIO
>>> from docling.datamodel.base_models import DocumentStream
>>> buf = BytesIO(b"<html><body>Hello</body></html>")
>>> stream = DocumentStream(name="page.html", stream=buf)
>>> result = converter.convert(stream)
```

#### convert_all

```python
convert_all(
    source: Iterable[Union[Path, str, DocumentStream]],
    headers: Optional[dict[str, str]] = None,
    raises_on_error: bool = True,
    max_num_pages: int = maxsize,
    max_file_size: int = maxsize,
    page_range: PageRange = DEFAULT_PAGE_RANGE
) -> Iterator[ConversionResult]
```

Convert multiple documents from file paths, URLs, or DocumentStreams.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `Iterable[Union[Path, str, DocumentStream]]` | *(required)* | Source of input documents given as an iterable of file paths, URLs, or DocumentStreams. |
| `headers` | `Optional[dict[str, str]]` | `None` | Optional headers given as a (single) dictionary of string key-value pairs, in case of URL input source. |
| `raises_on_error` | `bool` | `True` | Whether to raise an error on the first conversion failure. |
| `max_num_pages` | `int` | `maxsize` | Maximum number of pages accepted per document. Documents exceeding this number will not be converted. |
| `max_file_size` | `int` | `maxsize` | Maximum file size in bytes. Documents exceeding this limit will be skipped. |
| `page_range` | `PageRange` | `DEFAULT_PAGE_RANGE` | Range of pages to convert in each document. |

**Yields:**

| Type | Description |
|------|-------------|
| `ConversionResult` | The conversion results, each containing a `DoclingDocument` in the `document` attribute and metadata about the conversion process. |

**Raises:**

| Exception | Description |
|-----------|-------------|
| `ConversionError` | An error occurred during conversion. |

**Examples:**

Convert a batch of local files:

```python
>>> from pathlib import Path
>>> converter = DocumentConverter()
>>> paths = list(Path("docs/").glob("*.pdf"))
>>> for result in converter.convert_all(paths):
...     print(result.document.export_to_markdown()[:100])
```

Convert with a file size limit of 20 MB:

```python
>>> results = converter.convert_all(
...     paths, max_file_size=20 * 1024 * 1024
... )
```

#### convert_string

```python
convert_string(
    content: str,
    format: InputFormat,
    name: Optional[str] = None
) -> ConversionResult
```

Convert a document given as a string using the specified format.

Only Markdown (`InputFormat.MD`) and HTML (`InputFormat.HTML`) formats are supported. The content is wrapped in a `DocumentStream` and passed to the main conversion pipeline.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str` | *(required)* | The document content as a string. |
| `format` | `InputFormat` | *(required)* | The format of the input content. |
| `name` | `Optional[str]` | `None` | The filename to associate with the document. If not provided, a timestamp-based name is generated. The appropriate file extension (`md` or `html`) is appended if missing. |

**Returns:**

| Type | Description |
|------|-------------|
| `ConversionResult` | The conversion result, which contains a `DoclingDocument` in the `document` attribute, and metadata about the conversion process. |

**Raises:**

| Exception | Description |
|-----------|-------------|
| `ValueError` | If format is neither `InputFormat.MD` nor `InputFormat.HTML`. |
| `ConversionError` | An error occurred during conversion. |

**Examples:**

Convert a Markdown string:

```python
>>> from docling.datamodel.base_models import InputFormat
>>> converter = DocumentConverter()
>>> result = converter.convert_string(
...     "# Title\nSome text.", format=InputFormat.MD
... )
>>> print(result.document.export_to_markdown())
```

Convert an HTML string:

```python
>>> result = converter.convert_string(
...     "<h1>Title</h1><p>Some text.</p>",
...     format=InputFormat.HTML,
...     name="my_page",
... )
```

#### initialize_pipeline

```python
initialize_pipeline(format: InputFormat)
```

Initialize the conversion pipeline for the selected format.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `format` | `InputFormat` | *(required)* | The input format for which to initialize the pipeline. |

**Raises:**

| Exception | Description |
|-----------|-------------|
| `ConversionError` | If no pipeline could be initialized for the given format. |
| `RuntimeError` | If `artifacts_path` is set in `docling.datamodel.settings.settings` when required by the pipeline, but points to a non-directory file. |
| `FileNotFoundError` | If local model files are not found. |

---

## ConversionResult *(pydantic-model)*

Bases: `ConversionAssets`

### Attributes

| Field | Type | Kind |
|-------|------|------|
| `assembled` | `AssembledUnit` | pydantic-field |
| `confidence` | `ConfidenceReport` | pydantic-field |
| `document` | `DoclingDocument` | pydantic-field |
| `errors` | `list[ErrorItem]` | pydantic-field |
| `input` | `InputDocument` | pydantic-field |
| `legacy_document` | *(computed)* | property |
| `pages` | `list[Page]` | pydantic-field |
| `status` | `ConversionStatus` | pydantic-field |
| `timestamp` | `Optional[str]` | pydantic-field |
| `timings` | `dict[str, ProfilingItem]` | pydantic-field |
| `version` | `DoclingVersion` | pydantic-field |

### Methods

#### load

```python
load(filename: Union[str, Path]) -> ConversionAssets
```

Load a ConversionAssets. Class method.

#### save

```python
save(*, filename: Union[str, Path], indent: Optional[int] = 2)
```

Serialize the full ConversionAssets to JSON.

---

## ConversionStatus *(enum)*

Bases: `str`, `Enum`

| Value |
|-------|
| `FAILURE` |
| `PARTIAL_SUCCESS` |
| `PENDING` |
| `SKIPPED` |
| `STARTED` |
| `SUCCESS` |

---

## FormatOption *(pydantic-model)*

Bases: `BaseFormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[BackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type[BasePipeline]` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## InputFormat *(enum)*

Bases: `str`, `Enum`

A document format supported by document backend parsers.

| Value |
|-------|
| `ASCIIDOC` |
| `AUDIO` |
| `CSV` |
| `DOCX` |
| `HTML` |
| `IMAGE` |
| `JSON_DOCLING` |
| `LATEX` |
| `MD` |
| `METS_GBS` |
| `PDF` |
| `PPTX` |
| `VTT` |
| `XLSX` |
| `XML_JATS` |
| `XML_USPTO` |
| `XML_XBRL` |

---

## PdfFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[PdfBackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## ImageFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[BackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## StandardPdfPipeline

```python
StandardPdfPipeline(pipeline_options: ThreadedPdfPipelineOptions)
```

Bases: `ConvertPipeline`

High-performance PDF pipeline with multi-threaded stages.

### Attributes

| Field | Type | Kind |
|-------|------|------|
| `artifacts_path` | `Optional[Path]` | instance-attribute |
| `build_pipe` | `List[Callable]` | instance-attribute |
| `enrichment_pipe` | | instance-attribute |
| `keep_images` | | instance-attribute |
| `pipeline_options` | `ThreadedPdfPipelineOptions` | instance-attribute |

### Methods

#### execute

```python
execute(in_doc: InputDocument, raises_on_error: bool) -> ConversionResult
```

#### get_default_options

```python
get_default_options() -> ThreadedPdfPipelineOptions
```

Class method.

#### is_backend_supported

```python
is_backend_supported(backend: AbstractDocumentBackend) -> bool
```

Class method.

---

## WordFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[BackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## PowerpointFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[BackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## MarkdownFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[MarkdownBackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## AsciiDocFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[BackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## HTMLFormatOption *(pydantic-model)*

Bases: `FormatOption`

| Field | Type |
|-------|------|
| `backend` | `Type[AbstractDocumentBackend]` |
| `backend_options` | `Optional[HTMLBackendOptions]` |
| `model_config` | *(class-attribute)* |
| `pipeline_cls` | `Type` |
| `pipeline_options` | `Optional[PipelineOptions]` |

### Methods

#### set_optional_field_default

```python
set_optional_field_default() -> Self
```

Pydantic validator.

---

## SimplePipeline

```python
SimplePipeline(pipeline_options: ConvertPipelineOptions)
```

Bases: `ConvertPipeline`

SimpleModelPipeline. This class is used for formats/backends which produce straight DoclingDocument output.

### Attributes

| Field | Type | Kind |
|-------|------|------|
| `artifacts_path` | `Optional[Path]` | instance-attribute |
| `build_pipe` | `List[Callable]` | instance-attribute |
| `enrichment_pipe` | | instance-attribute |
| `keep_images` | | instance-attribute |
| `pipeline_options` | `ConvertPipelineOptions` | instance-attribute |

### Methods

#### execute

```python
execute(in_doc: InputDocument, raises_on_error: bool) -> ConversionResult
```

#### get_default_options

```python
get_default_options() -> ConvertPipelineOptions
```

Class method.

#### is_backend_supported

```python
is_backend_supported(backend: AbstractDocumentBackend)
```

Class method.
