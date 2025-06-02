import os
import sys
import json
import base64
from PIL import Image
from openai import OpenAI
import io
import cv2
import mimetypes


class Request:
    def __init__(self, model_name, prompt, image_urls, kwargs):
        self.model_name = model_name
        self.prompt = prompt
        self.image_urls = image_urls
        self.kwargs = kwargs

def encode_image_to_base64(path: str, quality: int = 20, max_size: int = 512) -> str:
    """
    Read local image, compress and return base64 encoded data URI string.
    Supports any image format, automatically infers MIME type.
    
    Args:
        path: Image file path
        quality: JPEG compression quality (1~100), default 20
        max_size: Maximum dimension (width or height) of the image, default 512
    
    Returns:
        str: Base64 encoded string in data URI format
    """
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        raise ValueError(f"Cannot recognize MIME type: {path}")

    # Read image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {path}")

    # Resize image if needed
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Compress to JPEG format to reduce size (regardless of original format)
    success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Image compression failed")

    # Encode to base64
    b64 = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    
    # Debug information
    original_size = os.path.getsize(path)
    compressed_size = len(encoded_img.tobytes())
    b64_size = len(b64)
    # print(f"Quality: {quality}")
    # print(f"Max size: {max_size}")
    # print(f"Original dimensions: {width}x{height}")
    # print(f"New dimensions: {img.shape[1]}x{img.shape[0]}")
    # print(f"Original file size: {original_size} bytes")
    # print(f"Compressed size: {compressed_size} bytes")
    # print(f"Base64 string length: {b64_size} characters")
    # print(f"Compression ratio: {compressed_size/original_size:.2f}")
    
    return f"data:image/jpeg;base64,{b64}"

def handle_qwen_request_sync(request):
    """
    Handle a synchronous Qwen request with image and text input.

    Args:
        request (object): Request object containing model_name, prompt, image_urls, and additional kwargs.

    Returns:
        str: The response text from the API.
    """
    # Initialize Qwen client
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't check key, can be EMPTY
        base_url="http://localhost:8005/v1",
    )

    # Prepare the image
    if isinstance(request.image_urls, str):
        image_urls = [request.image_urls]
    else:
        image_urls = request.image_urls

    # Encode the first image with reduced size
    img_b64 = encode_image_to_base64(image_urls[0], quality=20, max_size=512)  # 使用较小的分辨率

    # Prepare messages for the API request
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": request.prompt},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ],
    }]

    # Execute the request
    response = client.chat.completions.create(
        model="Qwen2.5-VL-32B-Instruct",
        messages=messages,
        max_completion_tokens=1024,
        **request.kwargs
    )
    
    response_text = response.choices[0].message.content

    if response_text:
        return response_text
    else:
        raise ValueError("Empty response from API")

def get_caption(image_urls, subject=True):
    if not isinstance(image_urls, (list, tuple)):
        image_urls = [image_urls]
    
    if subject:
        prompt = "Describe this image in detail using NO MORE than 20 words. Focus on the main subject in the image. Do not include any other unrelated information."
    else:
        prompt = "Describe this image in detail using NO MORE than 20 words. Do not include any other unrelated information."

    request = Request(
        model_name="Qwen2.5-VL-32B-Instruct",
        prompt=prompt,
        image_urls=image_urls,
        kwargs={"temperature": 0.7}
    )

    try:
        response = handle_qwen_request_sync(request)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    image_path = "/media/sata4/Contextaware/plot/frequency.png"
    caption = get_caption(image_path,subject=True)
    print(f"Generated caption: {caption}") 