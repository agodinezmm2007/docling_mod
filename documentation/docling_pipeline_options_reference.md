# Pipeline Options

Pipeline options allow to customize the execution of the models during the conversion pipeline.
This includes options for the OCR engines, the table model as well as enrichment options which
can be enabled with `do_xyz = True`.

This is an automatic generated API reference of all the pipeline options available in Docling.

## pipeline_options

### Classes

- `AsrPipelineOptions` — Configuration options for the Automatic Speech Recognition (ASR) pipeline.
- `BaseLayoutOptions` — Base options for document layout analysis models.
- `BaseOptions` — Base class for all pipeline option models.
- `BaseTableStructureOptions` — Base options for table structure extraction models.
- `CodeFormulaVlmOptions` — Configuration for VLM-based code and formula extraction.
- `ConvertPipelineOptions` — Base configuration for document conversion pipelines.
- `EasyOcrOptions` — Configuration for EasyOCR engine.
- `LayoutObjectDetectionOptions` — Options for layout detection using object-detection runtimes.
- `LayoutOptions` — Options for layout processing using Docling's built-in layout model.
- `OcrAutoOptions` — Automatic OCR engine selection based on system availability.
- `OcrEngine` — Available OCR engines for text extraction from images.
- `OcrMacOptions` — Configuration for native macOS OCR using Vision framework.
- `OcrOptions` — Base configuration for Optical Character Recognition engines.
- `PaginatedPipelineOptions` — Configuration for pipelines processing paginated documents.
- `PdfBackend` — Available PDF parsing backends for document processing.
- `PdfPipelineOptions` — Configuration options for the PDF document processing pipeline.
- `PictureDescriptionApiOptions` — Configuration for API-based picture description services.
- `PictureDescriptionBaseOptions` — Base configuration for picture description models.
- `PictureDescriptionVlmEngineOptions` — Configuration for VLM runtime-based picture description.
- `PictureDescriptionVlmOptions` — Configuration for inline vision-language models for picture description.
- `PipelineOptions` — Base configuration for document processing pipelines.
- `ProcessingPipeline` — Available document processing pipeline types for different use cases.
- `RapidOcrOptions` — Configuration for RapidOCR engine with multiple backend support.
- `TableFormerMode` — Operating modes for TableFormer table structure extraction model.
- `TableStructureOptions` — Options for the table structure (TableFormer V1).
- `TableStructureV2Options` — Options for the table structure (TableFormer V2).
- `TesseractCliOcrOptions` — Configuration for Tesseract OCR via command-line interface.
- `TesseractOcrOptions` — Configuration for Tesseract OCR via Python bindings (tesserocr).
- `ThreadedPdfPipelineOptions` — Pipeline options for the threaded PDF pipeline with batching and backpressure control.
- `VlmConvertOptions` — Configuration for VLM-based document conversion.
- `VlmExtractionPipelineOptions` — Options for VLM-based structured information extraction pipeline.
- `VlmPipelineOptions` — Pipeline configuration for vision-language model based document processing.

### Functions

- `normalize_pdf_backend` — Normalize deprecated backend enum values to current ones.

### Module Attributes

#### granite_picture_description

```python
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id='ibm-granite/granite-vision-3.3-2b',
    prompt='What is shown in this image?'
)
```

Pre-configured Granite Vision model options for picture description. Uses IBM's Granite Vision 3.3-2B model.

#### smolvlm_picture_description

```python
smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id='HuggingFaceTB/SmolVLM-256M-Instruct'
)
```

Pre-configured SmolVLM model options for picture description. Lightweight vision-language model optimized for generating descriptions of images.

---

## BaseOptions *(pydantic-model)*

Bases: `BaseModel`

Base class for all pipeline option models. Subclasses must declare a `kind` ClassVar that serves as a discriminator for polymorphic deserialization in Pydantic unions.

```python
kind: str  # class-attribute
```

---

## PipelineOptions *(pydantic-model)*

Bases: `BaseOptions`

Base configuration for document processing pipelines. Provides foundational settings shared by every pipeline type.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `document_timeout` | `Optional[float]` | `None` | Max processing time in seconds before aborting. Returns partial results with PARTIAL_SUCCESS. Recommended: 90-120s for production. |
| `accelerator_options` | `AcceleratorOptions` | `AcceleratorOptions()` | Hardware acceleration config: GPU device selection, memory management, execution optimization. |
| `enable_remote_services` | `bool` | `False` | Allow pipeline to call external APIs/cloud services. Required for API-based picture description. |
| `allow_external_plugins` | `bool` | `False` | Allow loading external third-party plugins for OCR, layout, table structure, or picture description. |
| `artifacts_path` | `Optional[Union[Path, str]]` | `None` | Local directory with pre-downloaded model artifacts. If None, models fetched from remote on first use. Use `docling-tools models download` to pre-fetch. |

### AcceleratorOptions

```python
class AcceleratorOptions(BaseModel):
    num_threads: int = 4            # CPU threads for inference. Set via DOCLING_NUM_THREADS.
    device: str = "auto"            # "auto", "cpu", "cuda", "cuda:N", "mps", "xpu". Set via DOCLING_DEVICE.
    cuda_use_flash_attention2: bool = False  # Flash Attention 2 for Ampere+ GPUs. Set via DOCLING_CUDA_USE_FLASH_ATTENTION2.
```

### AcceleratorDevice *(enum)*

| Value | Description |
|-------|-------------|
| `auto` | Automatic detection |
| `cpu` | CPU only |
| `cuda` | NVIDIA GPU |
| `mps` | Apple Silicon |
| `xpu` | Intel GPU |

---

## ConvertPipelineOptions *(pydantic-model)*

Bases: `PipelineOptions`

Adds picture classification, description, and chart extraction to the base pipeline options.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_picture_classification` | `bool` | `False` | Enable picture classification (photo, diagram, chart, etc.) |
| `picture_classification_options` | `DocumentPictureClassifierOptions` | *(defaults)* | Config for picture classification model/runtime. |
| `do_picture_description` | `bool` | `False` | Enable textual descriptions for pictures via VLMs. |
| `picture_description_options` | `PictureDescriptionBaseOptions` | *(defaults)* | Config for picture description model. Default: smolvlm preset. |
| `do_chart_extraction` | `bool` | `False` | Enable chart data extraction from bar, pie, line charts. |

*Inherits all fields from PipelineOptions.*

---

## PaginatedPipelineOptions *(pydantic-model)*

Bases: `ConvertPipelineOptions`

Adds page-level image generation controls for formats with discrete pages (PDF, PPTX, images).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `images_scale` | `float` | `1.0` | Scaling factor for generated images. 1.0 = standard, 2.0 = high res. |
| `generate_page_images` | `bool` | `False` | Generate PNG representations of each page. |
| `generate_picture_images` | `bool` | `False` | Extract and save embedded images as separate files. |

*Inherits all fields from ConvertPipelineOptions.*

---

## PdfPipelineOptions *(pydantic-model)*

Bases: `PaginatedPipelineOptions`

Full configuration for the PDF document processing pipeline.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_table_structure` | `bool` | `True` | Enable table structure extraction and reconstruction. |
| `do_ocr` | `bool` | `True` | Enable OCR for scanned/image-based PDFs. Significantly increases processing time. |
| `do_code_enrichment` | `bool` | `False` | Enable specialized code block processing. |
| `do_formula_enrichment` | `bool` | `False` | Enable mathematical formula recognition and LaTeX conversion. |
| `force_backend_text` | `bool` | `False` | Force PDF backend's native text extraction instead of layout model. |
| `table_structure_options` | `BaseTableStructureOptions` | `TableStructureOptions(mode='accurate')` | Table structure config. |
| `ocr_options` | `OcrOptions` | `OcrAutoOptions()` | OCR engine config. |
| `layout_options` | `BaseLayoutOptions` | `LayoutOptions()` | Layout analysis model config. Default: Heron. |
| `code_formula_options` | `CodeFormulaVlmOptions` | *(defaults)* | Code/formula extraction VLM config. |
| `generate_table_images` | `bool` | `False` | *(deprecated)* |
| `generate_parsed_pages` | `bool` | `False` | Retain intermediate parsed page representations. |
| `ocr_batch_size` | `int` | `4` | Batch size for OCR stage (threaded mode). |
| `layout_batch_size` | `int` | `4` | Batch size for layout analysis stage (threaded mode). |
| `table_batch_size` | `int` | `4` | Batch size for table structure stage (threaded mode). |
| `batch_polling_interval_seconds` | `float` | `0.5` | Polling interval for batch collection (threaded mode). |
| `queue_max_size` | `int` | `100` | Max queue size for inter-stage communication (threaded mode). |

*Inherits all fields from PaginatedPipelineOptions.*

**Notes:**
- Enabling multiple features (OCR, table structure, formulas) increases processing time significantly.
- For production systems, set `document_timeout` to 90-120 seconds.
- OCR requires system installation of engines (Tesseract, EasyOCR).
- RapidOCR has known issues with read-only filesystems (e.g., Databricks).

---

## ThreadedPdfPipelineOptions *(pydantic-model)*

Bases: `PdfPipelineOptions`

Pipeline options for the threaded PDF pipeline with batching and backpressure control. Processes pages through concurrent stages (OCR, layout, table structure) connected by bounded queues. Inherits all settings from `PdfPipelineOptions`.

---

## AsrPipelineOptions *(pydantic-model)*

Bases: `PipelineOptions`

Configuration for the Automatic Speech Recognition (ASR) pipeline. Processes audio files and converts speech to text using Whisper-based models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `asr_options` | `InlineAsrOptions` | *(defaults)* | ASR model config for audio transcription. Default: Whisper tiny. |

### InlineAsrOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_id` | `str` | *(required)* | HuggingFace model repo ID. Must be Whisper-compatible. |
| `verbose` | `bool` | `False` | Enable verbose logging. |
| `timestamps` | `bool` | `True` | Generate timestamps for transcribed segments. |
| `temperature` | `float` | `0.0` | Sampling temperature. 0.0 = greedy decoding. |
| `max_new_tokens` | `int` | `256` | Max tokens per transcription segment. |
| `max_time_chunk` | `float` | `30.0` | Max audio chunk duration in seconds. |
| `torch_dtype` | `Optional[str]` | `None` | PyTorch data type: `float32`, `float16`, `bfloat16`. |

---

## OcrOptions *(pydantic-model)*

Bases: `BaseOptions`

Base configuration for OCR engines.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | *(required)* | Language codes for OCR. Format depends on engine. |
| `force_full_page_ocr` | `bool` | `False` | Always apply full-page OCR. |
| `bitmap_area_threshold` | `float` | `0.05` | Min page area fraction for a bitmap to trigger OCR. |

---

## OcrAutoOptions *(pydantic-model)*

Bases: `OcrOptions`

Automatic OCR engine selection based on system availability. `lang` defaults to empty list (deferred to chosen engine).

```python
kind: Literal['auto'] = 'auto'
```

---

## OcrEngine *(enum)*

Bases: `str`, `Enum`

| Value | String | Description |
|-------|--------|-------------|
| `AUTO` | `'auto'` | Automatically select the best available OCR engine based on platform and installed libraries |
| `EASYOCR` | `'easyocr'` | Deep learning-based OCR supporting 80+ languages with GPU acceleration |
| `TESSERACT_CLI` | `'tesseract_cli'` | Tesseract OCR via command-line interface (requires system installation) |
| `TESSERACT` | `'tesseract'` | Tesseract OCR via Python bindings (tesserocr library) |
| `OCRMAC` | `'ocrmac'` | Native macOS Vision framework OCR (Apple platforms only) |
| `RAPIDOCR` | `'rapidocr'` | Lightweight OCR with multiple backend options (ONNX, OpenVINO, PaddlePaddle) |

---

## EasyOcrOptions *(pydantic-model)*

Bases: `OcrOptions`

```python
kind: Literal['easyocr'] = 'easyocr'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | `["fr", "de", "es", "en"]` | ISO 639-1 language codes. |
| `use_gpu` | `Optional[bool]` | `None` | Enable GPU. None = auto-detect. |
| `confidence_threshold` | `float` | `0.5` | Min confidence score for text recognition. |
| `model_storage_directory` | `Optional[str]` | `None` | Custom directory for downloaded models. |
| `recog_network` | `Optional[str]` | `"standard"` | Recognition network: `standard` or `craft`. |
| `download_enabled` | `bool` | `True` | Allow automatic model download. |
| `suppress_mps_warnings` | `bool` | `True` | Suppress MPS warnings on macOS. |
| `model_config` | | `ConfigDict(extra='forbid', protected_namespaces=())` |

---

## TesseractOcrOptions *(pydantic-model)*

Bases: `OcrOptions`

Configuration for Tesseract OCR via Python bindings (tesserocr).

```python
kind: Literal['tesserocr'] = 'tesserocr'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | `["fra", "deu", "spa", "eng"]` | ISO 639-2 3-letter codes. |
| `path` | `Optional[str]` | `None` | Tesseract data directory. None = default TESSDATA_PREFIX. |
| `psm` | `Optional[int]` | `None` | Page Segmentation Mode (0-13). Common: 3=auto, 6=uniform block, 11=sparse text. |
| `model_config` | | `ConfigDict(extra='forbid')` |

---

## TesseractCliOcrOptions *(pydantic-model)*

Bases: `OcrOptions`

Configuration for Tesseract OCR via command-line interface.

```python
kind: Literal['tesseract'] = 'tesseract'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | `["fra", "deu", "spa", "eng"]` | ISO 639-2 3-letter codes. |
| `tesseract_cmd` | `str` | `"tesseract"` | Command or path to Tesseract executable. |
| `path` | `Optional[str]` | `None` | Tesseract data directory. |
| `psm` | `Optional[int]` | `None` | Page Segmentation Mode (0-13). |
| `model_config` | | `ConfigDict(extra='forbid')` |

---

## OcrMacOptions *(pydantic-model)*

Bases: `OcrOptions`

Configuration for native macOS OCR using Vision framework.

```python
kind: Literal['ocrmac'] = 'ocrmac'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | `["fr-FR", "de-DE", "es-ES", "en-US"]` | Language-region locale codes. |
| `recognition` | `str` | `"accurate"` | `accurate` (higher quality, slower) or `fast`. |
| `framework` | `str` | `"vision"` | macOS framework. Currently: `vision`. |
| `model_config` | | `ConfigDict(extra='forbid')` |

---

## RapidOcrOptions *(pydantic-model)*

Bases: `OcrOptions`

Configuration for RapidOCR engine with multiple backend support.

```python
kind: Literal['rapidocr'] = 'rapidocr'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lang` | `list[str]` | `["english", "chinese"]` | Reserved for future compatibility. |
| `backend` | `Literal['onnxruntime', 'openvino', 'paddle', 'torch']` | `"onnxruntime"` | Inference backend. |
| `text_score` | `float` | `0.5` | Min confidence for text detection. |
| `use_det` | `Optional[bool]` | `None` | Enable text detection stage. |
| `use_cls` | `Optional[bool]` | `None` | Enable text direction classification. |
| `use_rec` | `Optional[bool]` | `None` | Enable text recognition. |
| `print_verbose` | `bool` | `False` | Verbose logging. |
| `det_model_path` | `Optional[str]` | `None` | Custom detection model path. |
| `cls_model_path` | `Optional[str]` | `None` | Custom classification model path. |
| `rec_model_path` | `Optional[str]` | `None` | Custom recognition model path. |
| `rec_keys_path` | `Optional[str]` | `None` | Custom recognition keys file path. |
| `rec_font_path` | `Optional[str]` | `None` | Custom recognition font path. |
| `font_path` | `Optional[str]` | `None` | Custom font file for visualization. |
| `rapidocr_params` | `dict[str, Any]` | `{}` | Additional pass-through parameters. |
| `model_config` | | `ConfigDict(extra='forbid')` |

---

## BaseLayoutOptions *(pydantic-model)*

Bases: `BaseOptions`

Base options for document layout analysis models. Layout analysis detects the structural regions of a document page (text blocks, tables, figures, headers, etc.) and assigns content cells to those regions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `keep_empty_clusters` | `bool` | `False` | Retain clusters without content. Enable for debugging. |
| `skip_cell_assignment` | `bool` | `False` | Skip cell-to-table assignment. Performance optimization. |

---

## LayoutOptions *(pydantic-model)*

Bases: `BaseLayoutOptions`

Options for Docling's built-in layout model.

```python
kind: str = 'docling_layout_default'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `create_orphan_clusters` | `bool` | `True` | Create clusters for orphaned elements. |
| `model_spec` | `LayoutModelConfig` | `DOCLING_LAYOUT_HERON` | Layout model. Options include HERON (default), EGRET_LARGE, EGRET_XLARGE. |

---

## LayoutObjectDetectionOptions *(pydantic-model)*

Bases: `ObjectDetectionStagePresetMixin`, `ObjectDetectionEngineOptionsMixin`, `BaseLayoutOptions`

Options for layout detection using object-detection runtimes. Alternative to `LayoutOptions` with preset support.

```python
kind: str = 'layout_object_detection'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine_options` | `BaseObjectDetectionEngineOptions` | *(required)* | Runtime config for object-detection engine. |
| `create_orphan_clusters` | `bool` | `False` | Create clusters for orphaned elements. |
| `model_spec` | `ObjectDetectionModelSpec` | *(from preset)* | Object-detection model spec. |

**Methods:** `from_preset(preset_id, engine_options=None, **overrides)`, `get_preset(preset_id)`, `list_preset_ids()`, `list_presets()`, `register_preset(preset)`, `get_preset_info()`

---

## BaseTableStructureOptions *(pydantic-model)*

Bases: `BaseOptions`

Base for all table structure backends.

---

## TableStructureOptions *(pydantic-model)*

Bases: `BaseTableStructureOptions`

Options for TableFormer V1.

```python
kind: str = 'docling_tableformer'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_cell_matching` | `bool` | `True` | Align detected cells with content. |
| `mode` | `TableFormerMode` | `"accurate"` | `accurate` (higher quality) or `fast` (speed). |

---

## TableStructureV2Options *(pydantic-model)*

Bases: `BaseTableStructureOptions`

Options for TableFormer V2.

```python
kind: str = 'docling_tableformer_v2'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_cell_matching` | `bool` | `True` | Align detected cells with content. |

---

## TableFormerMode *(enum)*

Bases: `str`, `Enum`

| Value | String | Description |
|-------|--------|-------------|
| `FAST` | `'fast'` | Speed over precision. For simple tables or high-volume processing. |
| `ACCURATE` | `'accurate'` | Higher quality, slower. Recommended for complex tables and production. |

---

## PdfBackend *(enum)*

Bases: `str`, `Enum`

| Value | String | Description |
|-------|--------|-------------|
| `PYPDFIUM2` | `'pypdfium2'` | Standard parser using PyPDFium2. Fast and reliable for basic extraction. |
| `DOCLING_PARSE` | `'docling_parse'` | Enhanced layout analysis, structure preservation, advanced table detection. Recommended. |
| `DLPARSE_V1` | `'dlparse_v1'` | Deprecated. Maps to DOCLING_PARSE. |
| `DLPARSE_V2` | `'dlparse_v2'` | Deprecated. Maps to DOCLING_PARSE. |
| `DLPARSE_V4` | `'dlparse_v4'` | Deprecated. Maps to DOCLING_PARSE. |

---

## ProcessingPipeline *(enum)*

Bases: `str`, `Enum`

| Value | String | Description |
|-------|--------|-------------|
| `LEGACY` | `'legacy'` | Legacy pipeline for backward compatibility. |
| `STANDARD` | `'standard'` | General document processing (PDF, DOCX, images) with layout analysis. |
| `VLM` | `'vlm'` | Vision-Language Model pipeline for advanced document understanding. |
| `ASR` | `'asr'` | Automatic Speech Recognition pipeline for audio/video transcription. |

---

## PictureDescriptionBaseOptions *(pydantic-model)*

Bases: `BaseOptions`

Base configuration for picture description models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `8` | Images per batch. Higher = more throughput, more memory. |
| `scale` | `float` | `2.0` | Image resolution scaling factor. Range: 0.5-4.0. |
| `picture_area_threshold` | `float` | `0.05` | Min picture area as fraction of page area (0.0-1.0). |
| `classification_allow` | `Optional[list[PictureClassificationLabel]]` | `None` | Only describe pictures with these labels. None = all allowed. |
| `classification_deny` | `Optional[list[PictureClassificationLabel]]` | `None` | Skip pictures with these labels. |
| `classification_min_confidence` | `float` | `0.0` | Min classification confidence to process (0.0-1.0). |

---

## PictureDescriptionApiOptions *(pydantic-model)*

Bases: `PictureDescriptionBaseOptions`

Configuration for API-based picture description. Sends images to an OpenAI-compatible chat completions endpoint.

```python
kind: Literal['api'] = 'api'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `AnyUrl` | `"http://localhost:8000/v1/chat/completions"` | API endpoint URL. |
| `headers` | `dict[str, str]` | `{}` | HTTP headers (e.g., `{"Authorization": "Bearer TOKEN"}`). |
| `params` | `dict[str, Any]` | `{}` | Additional query parameters. |
| `timeout` | `float` | `20.0` | Max seconds to wait for API response. |
| `concurrency` | `int` | `1` | Number of concurrent API requests. |
| `prompt` | `str` | `"Describe this image in a few sentences."` | Prompt template. |
| `provenance` | `str` | `""` | Provenance metadata for tracking. |

**Note:** Requires `enable_remote_services=True` on the parent pipeline.

---

## PictureDescriptionVlmOptions *(pydantic-model)*

Bases: `PictureDescriptionBaseOptions`

Legacy implementation using direct HuggingFace Transformers integration. For the new system, use `PictureDescriptionVlmEngineOptions`.

```python
kind: Literal['vlm'] = 'vlm'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_id` | `str` | *(required)* | HuggingFace model repo ID for the VLM. |
| `prompt` | `str` | `"Describe this image in a few sentences."` | Prompt template. |
| `generation_config` | `dict[str, Any]` | `{"max_new_tokens": 200, "do_sample": False}` | HuggingFace generation config. |

**Property:** `repo_cache_folder` — Local cache folder name derived from `repo_id` (replaces `/` with `--`).

---

## PictureDescriptionVlmEngineOptions *(pydantic-model)*

Bases: `StagePresetMixin`, `VlmEngineOptionsMixin`, `PictureDescriptionBaseOptions`

New implementation using pluggable runtime system with preset support. Supports Transformers, MLX, API, etc.

```python
kind: Literal['picture_description_vlm_engine'] = 'picture_description_vlm_engine'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine_options` | `BaseVlmEngineOptions` | *(required)* | Runtime config (transformers, mlx, api, etc.) |
| `model_spec` | `VlmModelSpec` | *(required)* | Model specification with runtime-specific overrides. |
| `prompt` | `str` | `"Describe this image in a few sentences."` | Prompt template. |
| `generation_config` | `dict[str, Any]` | `{"max_new_tokens": 200, "do_sample": False}` | Generation config. |

**Examples:**

```python
# Use preset with default runtime
options = PictureDescriptionVlmEngineOptions.from_preset("smolvlm")

# Use preset with runtime override
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions, VlmEngineType
options = PictureDescriptionVlmEngineOptions.from_preset(
    "smolvlm",
    engine_options=MlxVlmEngineOptions(engine_type=VlmEngineType.MLX)
)
```

**Methods:** `from_preset(preset_id, engine_options=None, **overrides)`, `get_preset(preset_id)`, `list_preset_ids()`, `list_presets()`, `register_preset(preset)`, `get_preset_info()`, `resolve_engine_options(value)`

---

## CodeFormulaVlmOptions *(pydantic-model)*

Bases: `StagePresetMixin`, `VlmEngineOptionsMixin`, `BaseModel`

Configuration for VLM-based code and formula extraction.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine_options` | `BaseVlmEngineOptions` | *(required)* | Runtime config. |
| `model_spec` | `VlmModelSpec` | *(required)* | Model specification. |
| `scale` | `float` | `2.0` | Image scaling factor. |
| `max_size` | `Optional[int]` | `None` | Max image dimension (width or height). |
| `extract_code` | `bool` | `True` | Extract code blocks. |
| `extract_formulas` | `bool` | `True` | Extract mathematical formulas. |

**Examples:**

```python
options = CodeFormulaVlmOptions.from_preset("codeformulav2")
options = CodeFormulaVlmOptions.from_preset("granite_docling")
```

**Methods:** `from_preset(preset_id, engine_options=None, **overrides)`, `get_preset(preset_id)`, `list_preset_ids()`, `list_presets()`, `register_preset(preset)`, `get_preset_info()`, `resolve_engine_options(value)`

---

## VlmConvertOptions *(pydantic-model)*

Bases: `StagePresetMixin`, `VlmEngineOptionsMixin`, `BaseModel`

Configuration for VLM-based document conversion. Converts document pages to structured formats (DocTags, Markdown, etc.).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine_options` | `BaseVlmEngineOptions` | *(required)* | Runtime config. |
| `model_spec` | `VlmModelSpec` | *(required)* | Model specification. |
| `scale` | `float` | `2.0` | Image scaling factor. |
| `max_size` | `Optional[int]` | `None` | Max image dimension. |
| `batch_size` | `int` | `1` | Batch size for processing pages. |
| `force_backend_text` | `bool` | `False` | Force backend text extraction instead of VLM. |

**Examples:**

```python
# Use preset with default runtime
options = VlmConvertOptions.from_preset("smoldocling")

# Use preset with API runtime override
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
options = VlmConvertOptions.from_preset(
    "smoldocling",
    engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API_OLLAMA)
)
```

**Methods:** `from_preset(preset_id, engine_options=None, **overrides)`, `get_preset(preset_id)`, `list_preset_ids()`, `list_presets()`, `register_preset(preset)`, `get_preset_info()`, `resolve_engine_options(value)`

---

## VlmPipelineOptions *(pydantic-model)*

Bases: `PaginatedPipelineOptions`

Pipeline configuration for VLM-based document processing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vlm_options` | `Union[VlmConvertOptions, InlineVlmOptions, ApiVlmOptions]` | *(required)* | VLM conversion options. |
| `force_backend_text` | `bool` | `False` | Force backend text extraction. |

*Inherits all fields from PaginatedPipelineOptions.*

---

## VlmExtractionPipelineOptions *(pydantic-model)*

Bases: `PipelineOptions`

Options for VLM-based structured information extraction pipeline.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vlm_options` | `InlineVlmOptions` | *(required)* | VLM conversion options. |

*Inherits all fields from PipelineOptions.*

---

## VlmModelSpec *(pydantic-model)*

Specification for a VLM model, independent of the engine.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable model name. |
| `default_repo_id` | `str` | *(required)* | Default HuggingFace repository ID. |
| `revision` | `str` | `"main"` | Model revision. |
| `prompt` | `str` | *(required)* | Prompt template. |
| `response_format` | `ResponseFormat` | *(required)* | Expected response format. |
| `supported_engines` | `Optional[list[VlmEngineType]]` | `None` | Supported engines (None = all). |
| `engine_overrides` | `dict[VlmEngineType, EngineModelConfig]` | `{}` | Engine-specific config overrides. |
| `api_overrides` | `dict[VlmEngineType, ApiModelConfig]` | `{}` | API-specific config overrides. |
| `trust_remote_code` | `bool` | `False` | Trust remote code for this model. |
| `stop_strings` | `list[str]` | `[]` | Stop strings for generation. |
| `max_new_tokens` | `int` | `4096` | Max new tokens to generate. |

---

## VlmEngineType *(enum)*

| Value | Description |
|-------|-------------|
| `transformers` | HuggingFace Transformers |
| `mlx` | Apple MLX framework |
| `vllm` | vLLM inference server |
| `api` | Generic OpenAI-compatible API |
| `api_ollama` | Ollama API |
| `api_lmstudio` | LM Studio API |
| `api_openai` | OpenAI API |
| `auto_inline` | Auto-select best available inline engine |

---

## ResponseFormat *(enum)*

| Value |
|-------|
| `doctags` |
| `markdown` |
| `deepseekocr_markdown` |
| `html` |
| `otsl` |
| `plaintext` |

---

## PictureClassificationLabel *(enum)*

| Value |
|-------|
| `other`, `picture_group`, `pie_chart`, `bar_chart`, `stacked_bar_chart`, `line_chart`, `flow_chart`, `scatter_chart`, `heatmap`, `remote_sensing`, `natural_image`, `chemistry_molecular_structure`, `chemistry_markush_structure`, `icon`, `logo`, `signature`, `stamp`, `qr_code`, `bar_code`, `screenshot`, `map`, `stratigraphic_chart`, `cad_drawing`, `electrical_diagram` |

---

## normalize_pdf_backend

```python
def normalize_pdf_backend(backend: PdfBackend) -> PdfBackend
```

Normalize deprecated backend enum values to current ones. Maps `DLPARSE_V1`, `DLPARSE_V2`, `DLPARSE_V4` to `DOCLING_PARSE`.
