# code_formula_model_vllm_api.py
# modified to use VLLM API endpoint on port 8006 instead 
# sends formula/code snippets to granite-vision-3.3-2b via API

import re
import logging
import base64
import requests
from io import BytesIO
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    TextItem,
    BoundingBox,
    DocItem,
)
from docling_core.types.doc.labels import CodeLanguageLabel
from PIL import Image, ImageOps
from pydantic import BaseModel

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BaseItemAndImageEnrichmentModel

_log = logging.getLogger(__name__)


class CodeFormulaModelOptions(BaseModel):
    """
    Configuration options for the VLLM API-based CodeFormulaModel.
    """
    kind: Literal["vllm_api"] = "vllm_api"
    do_code_enrichment: bool = True
    do_formula_enrichment: bool = True
    api_url: str = "http://localhost:8006/v1/chat/completions"
    api_timeout: int = 120


class CodeFormulaModel(BaseItemAndImageEnrichmentModel):
    """
    Model using VLLM API endpoint for code/formula recognition.
    Sends snippet images to granite-vision-3.3-2b running on port 8006.
    """

    elements_batch_size = 5
    images_scale = 2.6
    expansion_factor = 0.18
    right_expansion_factor = 0.03
    left_expansion_factor = 0.04

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CodeFormulaModelOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """
        Initializes the VLLM API-based CodeFormulaModel.
        No local model loading - all requests go to API endpoint.
        """
        super().__init__()
        self.enabled = enabled
        self.options = options
        self.api_url = options.api_url
        self.api_timeout = options.api_timeout

        _log.info(
            "Initialized VLLM API CodeFormulaModel. API URL: %s, Timeout: %ds",
            self.api_url, self.api_timeout
        )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Returns True if the given doc element is a code snippet or a formula
        and if this model is enabled for those tasks.
        """
        if not self.enabled:
            return False

        # check if it's a code snippet item
        if isinstance(element, CodeItem) and self.options.do_code_enrichment:
            return True

        # check if it's a formula-labeled text item
        if (
            isinstance(element, TextItem)
            and element.label == DocItemLabel.FORMULA
            and self.options.do_formula_enrichment
        ):
            return True

        return False

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[ItemAndImageEnrichmentElement]:
        """
        Prepares element with asymmetric bbox expansion and masked images for formulas.
        """
        if not self.is_processable(doc=conv_res.document, element=element):
            return None

        assert isinstance(element, DocItem)
        element_prov = element.prov[0]

        bbox = element_prov.bbox
        width = bbox.r - bbox.l
        height = bbox.t - bbox.b

        # asymmetric bounding box expansion
        expanded_bbox = BoundingBox(
            l=bbox.l - width * self.left_expansion_factor,
            t=bbox.t + height * self.expansion_factor,
            r=bbox.r + width * self.right_expansion_factor,
            b=bbox.b - height * self.expansion_factor,
            coord_origin=bbox.coord_origin,
        )

        page_ix = element_prov.page_no - 1
        page = conv_res.pages[page_ix]

        # use masked image for formulas, regular for code
        if element.label == DocItemLabel.FORMULA:
            pdf_identifier = conv_res.input.file.stem
            cropped_image = page.get_masked_image(
                scale=self.images_scale, cropbox=expanded_bbox, pdf_identifier=pdf_identifier
            )
        else:
            cropped_image = page.get_image(
                scale=self.images_scale, cropbox=expanded_bbox
            )

        return ItemAndImageEnrichmentElement(item=element, image=cropped_image)

    def _extract_code_language(self, input_string: str) -> Tuple[str, Optional[str]]:
        """
        Extracts language tag from model output.
        Pattern: <_Language_> content
        """
        pattern = r"^<_([^_>]+)_>\s*(.*)"
        match = re.match(pattern, input_string, flags=re.DOTALL)
        if match:
            language = str(match.group(1))
            remainder = str(match.group(2))
            return remainder, language
        else:
            return input_string, None

    def _get_code_language_enum(self, value: Optional[str]) -> CodeLanguageLabel:
        """
        Convert a string to CodeLanguageLabel enum or fallback to UNKNOWN.
        """
        if not isinstance(value, str):
            return CodeLanguageLabel.UNKNOWN

        try:
            return CodeLanguageLabel(value)
        except ValueError:
            return CodeLanguageLabel.UNKNOWN

    def _get_prompt(self, label: str) -> str:
        """
        Constructs prompt for the VLM API based on the input label.
        """
        if label == "code" or label == DocItemLabel.CODE:
            prompt_text = "Convert this code snippet to text. Identify the programming language and output the code."
        elif label == "formula" or label == DocItemLabel.FORMULA:
            prompt_text = "Convert this mathematical formula to LaTeX. Output only the LaTeX code without explanations or delimiters."
        else:
            prompt_text = "Convert this image to text."

        return prompt_text

    def _get_most_frequent_edge_color(self, pil_img: Image.Image) -> Union[int, Tuple[int, int, int]]:
        """
        Compute the most frequent color along the edges of a PIL image.
        Works for both grayscale and color images.
        """
        img_np = np.array(pil_img)
        if img_np.ndim == 2:  # grayscale
            top = img_np[0, :]
            bottom = img_np[-1, :]
            left = img_np[:, 0]
            right = img_np[:, -1]
            edges = np.concatenate([top, bottom, left, right])
            freq = Counter(edges.tolist())
            most_common_value, _ = freq.most_common(1)[0]
            return int(most_common_value)
        else:  # color
            top = img_np[0, :, :]
            bottom = img_np[-1, :, :]
            left = img_np[:, 0, :]
            right = img_np[:, -1, :]
            edges = np.concatenate([top, bottom, left, right], axis=0)
            edges_as_tuples = [tuple(pixel) for pixel in edges]
            freq = Counter(edges_as_tuples)
            most_common_value, _ = freq.most_common(1)[0]
            return most_common_value

    def _pad_with_most_frequent_edge_color(
        self,
        img: Union[Image.Image, np.ndarray],
        pad_top_ratio: float = 0.13,
        pad_bottom_ratio: float = 0.13,
        pad_left_ratio: float = 0.03,
        pad_right_ratio: float = 0.03
    ) -> Image.Image:
        """
        Pads an image with a border whose size is determined by a fixed ratio of the image dimensions.
        The border color is chosen as the most frequent edge color.
        """
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img.convert("RGB")

        # compute dynamic padding amounts based on the image dimensions
        width, height = pil_img.size
        pad_top = int(height * pad_top_ratio)
        pad_bottom = int(height * pad_bottom_ratio)
        pad_left = int(width * pad_left_ratio)
        pad_right = int(width * pad_right_ratio)

        most_freq_color = self._get_most_frequent_edge_color(pil_img)
        padded_img = ImageOps.expand(pil_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=most_freq_color)
        return padded_img

    def _extract_latex_from_response(self, response_text: str) -> str:
        """
        Extract LaTeX formula from VLM response.
        Removes <formula> tags and location markers if present.
        """
        # remove <formula> tags
        text = re.sub(r'<formula>.*?</formula>', '', response_text, flags=re.DOTALL)

        # remove location markers
        text = re.sub(r'<loc_\d+>', '', text)

        # extract content between $$ or $ delimiters if present
        dollar_match = re.search(r'\$\$(.*?)\$\$', text, re.DOTALL)
        if dollar_match:
            return dollar_match.group(1).strip()

        dollar_match = re.search(r'\$(.*?)\$', text)
        if dollar_match:
            return dollar_match.group(1).strip()

        # if no delimiters, return cleaned text
        return text.strip()

    def _send_to_vllm_api(self, img: Image.Image, prompt: str, snippet_id: str) -> Optional[str]:
        """
        Send image to VLLM API endpoint and get response.
        """
        try:
            # convert image to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # create API request payload
            payload = {
                "model": "ibm-granite/granite-vision-3.3-2b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.0
            }

            _log.debug(f"Sending snippet {snippet_id} to VLLM API at {self.api_url}")

            # send request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.api_timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                _log.debug(f"Snippet {snippet_id} API response: {content[:200]}...")
                return content
            else:
                _log.error(f"API request failed for snippet {snippet_id}: HTTP {response.status_code}")
                return None

        except Exception as e:
            _log.error(f"Error sending snippet {snippet_id} to API: {e}", exc_info=True)
            return None

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """
        Processes batch of elements using VLLM API for formula/code extraction.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images, labels, items, snippet_ids = [], [], [], []

        # collect and preprocess elements
        for idx, el in enumerate(element_batch):
            try:
                # only process text-based snippets
                if not isinstance(el.item, TextItem):
                    yield el.item
                    continue

                if isinstance(el.item, CodeItem):
                    labels.append("code")
                elif isinstance(el.item, TextItem) and el.item.label == DocItemLabel.FORMULA:
                    labels.append("formula")
                else:
                    continue

                # apply ratio-based padding
                try:
                    padded = self._pad_with_most_frequent_edge_color(el.image)
                except Exception as e:
                    _log.error("Error during padding for element %s: %s", el, e, exc_info=True)
                    continue

                images.append(padded)
                items.append(el.item)
                snippet_ids.append(f"{doc.name}_item_{idx}")
            except Exception as e:
                _log.error("Error processing element %s: %s", el, e, exc_info=True)
                continue

        if not images:
            return

        _log.info("Starting VLM API prediction for %d images with labels: %s", len(images), labels)

        # process each snippet via API
        for i, (img, lbl, item, snippet_id) in enumerate(zip(images, labels, items, snippet_ids)):
            try:
                # get prompt for this label
                prompt = self._get_prompt(lbl)

                # send to VLM API
                api_response = self._send_to_vllm_api(img, prompt, snippet_id)

                if not api_response:
                    _log.warning(f"Snippet {snippet_id} got no response from API. Skipping.")
                    continue

                # extract recognized text based on label
                if lbl == "formula":
                    # extract LaTeX from response
                    recognized_str = self._extract_latex_from_response(api_response)

                    # skip malformed results
                    if not recognized_str or recognized_str == "MALFORMED_FORMULA":
                        _log.warning(f"Snippet {snippet_id} produced malformed or empty output. Skipping.")
                        continue

                    item.text = recognized_str
                    _log.info(f"Snippet {snippet_id} extracted formula: {recognized_str[:100]}...")

                elif lbl == "code" and isinstance(item, CodeItem):
                    # for code, try to extract language and content
                    recognized_str, code_language = self._extract_code_language(api_response)
                    item.code_language = self._get_code_language_enum(code_language)
                    item.text = recognized_str
                    _log.info(f"Snippet {snippet_id} extracted code ({code_language}): {recognized_str[:100]}...")

                yield item

            except Exception as e:
                _log.error("Error processing snippet %s: %s; skipping item.", snippet_id, e, exc_info=True)
                continue

        _log.info("All done. Returning VLM API recognized results.")
