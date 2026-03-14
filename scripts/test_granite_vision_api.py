
"""
Test Granite Vision API using Docling's exact API request format.
Sends image to port 8006 (docling-granite-vision VLLM server).
"""
import base64
import json
from io import BytesIO
from PIL import Image
import requests

# Image path
IMAGE_PATH = "/mnt/c/Users/WSTATION/Desktop/docling_mods/data/figures/ijerph-19-04851_cleaned_fig4.png"

# API endpoint (port 8006 -> 8000 in container)
API_URL = "http://localhost:8006/v1/chat/completions"

# Step 1: Load image
print("Step 1: Loading image...")
image = Image.open(IMAGE_PATH)
print(f"  Image size: {image.size}")
print(f"  Image mode: {image.mode}")

# Step 2: Convert image to PNG and base64 encode (exactly as Docling does)
print("\nStep 2: Converting to PNG and base64 encoding...")
img_io = BytesIO()
image.save(img_io, "PNG")
image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
print(f"  Base64 length: {len(image_base64)} characters")

# Step 3: Create OpenAI-compatible messages (Docling format)
print("\nStep 3: Creating messages payload...")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
            {
                "type": "text",
                "text": "Describe this image in detail, including any text, labels, and key features.",
            },
        ],
    }
]

# Step 4: Create request payload
payload = {
    "messages": messages,
    "model": "ibm-granite/granite-vision-3.3-2b",
    "max_tokens": 1500,
    "temperature": 0.0,
}

print(f"  Payload keys: {list(payload.keys())}")
print(f"  Model: {payload['model']}")
print(f"  Max tokens: {payload['max_tokens']}")
print(f"  Temperature: {payload['temperature']}")

# Step 5: Send POST request to VLLM server
print(f"\nStep 4: Sending POST request to {API_URL}...")
try:
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    print(f"  Status code: {response.status_code}")

    if response.ok:
        print("\nStep 5: Parsing response...")
        result = response.json()

        # Extract description from OpenAI-compatible response
        description = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})

        print("\n" + "="*80)
        print("GRANITE VISION DESCRIPTION:")
        print("="*80)
        print(description)
        print("="*80)
        print(f"\nTokens used: {usage.get('total_tokens', 'N/A')}")
        print(f"Finish reason: {result['choices'][0].get('finish_reason', 'N/A')}")
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\nException: {e}")
