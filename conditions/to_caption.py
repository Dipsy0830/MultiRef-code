import os
import sys
import json
import base64
from PIL import Image
from openai import OpenAI, AzureOpenAI  # Import necessary modules
import io
from huggingface_hub import hf_hub_download

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
config=read_json('/media/sata4/Contextaware/config.json')

class Request:
    def __init__(self, model_name, prompt, image_urls, kwargs):
        self.model_name = model_name
        self.prompt = prompt
        self.image_urls = image_urls
        self.kwargs = kwargs

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def check_image_format(image_path):
    with open(image_path, 'rb') as f:
        file_header = f.read(12) 
    if file_header.startswith(b'\x89PNG\r\n\x1a\n'):
        return "png"
    elif file_header.startswith(b'\xFF\xD8'):
        return "jpeg"
    elif file_header[:4] == b'RIFF' and file_header[8:12] == b'WEBP':
        return "webp"
    else:
        return "unknown"


def handle_openai_request_sync(config, request):
    """
    Handle a synchronous OpenAI or Azure OpenAI request with image and text input.

    Args:
        config (dict): Configuration dictionary containing API keys, URLs, and Azure settings.
        request (object): Request object containing model_name, prompt, image_urls, and additional kwargs.

    Returns:
        str: The response text from the API.
    """

    def generate_image_messages(image_urls,low_detail=True):
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        image_messages = []
        for image_url in image_urls:
            image = Image.open(image_url)
            base64_image = encode_image_to_base64(image)
            img_format=check_image_format(image_url)
            image_message = {"type": "image_url", 
                             "image_url": {"url": f"data:image/{img_format};base64,{base64_image}",},
                            }
            if low_detail:
                image_message["image_url"]["detail"] = "low"
            image_messages.append(image_message)
            
            
        return image_messages
            
    # Determine SDK type and initialize the client
    sdk_type = config['openai_sdk_vlms'][request.model_name]
    if sdk_type == "AZURE":
        args={
            'api_key':config['AZURE']['AZURE_API_KEY'],
            "api_version":config['AZURE']['AZURE_API_VERSION'],
            "azure_endpoint":config['AZURE']['AZURE_ENDPOINT']
        }
        client = AzureOpenAI(**args)
    elif sdk_type == "OPENAI":
        client = OpenAI(
            api_key=config['OPENAI']['OPENAI_API_KEY'],
            base_url=config['OPENAI']['OPENAI_BASE_URL']
        )
    else:
        raise ValueError(f"Unsupported SDK type: {sdk_type}")

    image_messages = generate_image_messages(request.image_urls)
    base64_image=encode_image_to_base64(Image.open(request.image_urls[0]))
    # Prepare parameters for the API request
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze images and provide detailed descriptions."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                *image_messages,
            ]
        }
    ]

    parameters = {
        "model": request.model_name,
        "messages": messages,
        **request.kwargs,
    }

    # Execute the request
    response = client.chat.completions.create(**parameters)
    response_text = response.choices[0].message.content

    if response_text:
        return response_text
    else:
        raise ValueError("Empty response from API")


def get_caption(image_urls,subject=True):
    if not isinstance(image_urls, (list, tuple)):
        image_urls = [image_urls]
    
    if subject:
        prompt="Describe this image in detail using no more than 20 words. Focus on the main subject in the image. Do not include any other unrelated information."
    else:
        prompt="Describe this image in detail using no more than 20 words.  Do not include any other unrelated information."

    # Simulate the request object
    # request = Request(
    #     model_name="gpt-4o-mini-s3",
    #     prompt=prompt,
    #     image_urls=image_urls,
    #     kwargs={"temperature": 0.7, "max_tokens": 300}
    # )

    request = Request(
        model_name="gpt-4o-mini",
        prompt=prompt,
        image_urls=image_urls,
        kwargs={"temperature": 0.7, "max_tokens": 300}
    )

    # Call the request handler
    try:
        response = handle_openai_request_sync(config, request)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error occurred: {e}")
        return None
