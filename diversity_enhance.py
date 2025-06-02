
import os
import json
import tempfile
from openai import AzureOpenAI,OpenAI
import time
from collections import OrderedDict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import random

def build_prompt(basic_instruction, persona_list):
    """
    Constructs a prompt based on the given instruction and a randomly selected persona.

    Args:
        basic_instruction (str): The main text that needs to be refined or rewritten.
        persona_list (list): A list of personas to choose from.

    Returns:
        list: A list of dictionaries representing the chat messages for the OpenAI API.
    """
    selected_persona = random.choice(persona_list)
    prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in language refinement and creativity."
        },
        {
            "role": "user",
            "content": (
                f"You will adopt the persona of {selected_persona}. You will be given a text and your task is to rewrite "
                f"and polish it in a more diverse and creative manner that reflects the persona's style. Do not include "
                f"any direct references to the persona itself.\n"
                f"- You may alter sentence structure, wording, and tone.\n"
                f"- Do not modify text enclosed in angle brackets '< >'.\n"
                f"- If there is a 'caption:' section in the text, do not change anything following 'caption:'.\n\n"
                f"Here is the text:\n{basic_instruction}\n"
                f"Please provide the revised text directly without any additional commentary."
            )
        }
    ]
    return prompt

def get_response_gpt(basic_instruction, persona_list,config):
    
    client = OpenAI(
        api_key=config["OPENAI"]["OPENAI_API_KEY"],
        base_url=config["OPENAI"]["OPENAI_BASE_URL"]
    )
    model = "gpt-4o-mini"
    # 调用独立的 prompt 构建函数
    message = build_prompt(basic_instruction, persona_list)
    
    chat_completion = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0.7,
    )
    full_response = chat_completion.choices[0].message.content
    return full_response



def get_response_with_retry(prompt,persona_list, max_retries=3, retry_delay=1,config=None):
    """
    Get response with retry mechanism

    Args:
        prompt: Input prompt for getting response
        max_retries: Maximum number of retry attemptsNone
        retry_delay: Delay between retries in seconds

    Returns:
        Response from the API
    """
    for attempt in range(max_retries):
        try:
            response = get_response_gpt(prompt,persona_list,config)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"API call failed after {max_retries} attempts: {str(e)}")
                raise
            
