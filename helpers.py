import numpy as np
import os
from typing import List, Dict, Any
from vllm import LLM
from vllm.multimodal.utils import fetch_image
from PIL import Image
from jinja2 import Template
from pathlib import Path

def format_input_to_conversation(
    input_dict: Dict[str, Any], 
    default_instruction: str = "Represent the user's input."
) -> List[Dict]:
    content = []
    
    instruction = input_dict.get('instruction') or default_instruction
    text = input_dict.get('text')
    image = input_dict.get('image')
    
    if image:
        image_content = None
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = 'file://' + abs_image_path
        else:
            image_content = image
        
        if image_content:
            content.append({
                'type': 'image', 
                'image': image_content,
            })
    
    if text:
        content.append({'type': 'text', 'text': text})
    
    if not content:
        content.append({'type': 'text', 'text': ""})
    
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content}
    ]
    
    return conversation

def prepare_vllm_inputs_embedding(
    input_dict: Dict[str, Any], 
    llm, 
) -> Dict[str, Any]:
    conversation = format_input_to_conversation(input_dict)
    
    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    multi_modal_data = None
    image = input_dict.get('image')
    if image:
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    image_obj = Image.open(abs_image_path)
                    multi_modal_data = {"image": image_obj}
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            multi_modal_data = {"image": image}
    
    result = {
        "prompt": prompt_text,
        "multi_modal_data": multi_modal_data
    }
    return result


def parse_input_dict(input_dict: Dict[str, Any]):
    """
    Parse input dictionary to extract image and text content.
    Returns the formatted content string and multimodal data.
    """
    image = input_dict.get('image')
    text = input_dict.get('text')

    mm_data = {
        'image': []
    }
    content = ''
    if image:
        content += '<|vision_start|><|image_pad|><|vision_end|>'
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                try:
                    image_obj = fetch_image(image)
                    mm_data['image'].append(image_obj)
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    from PIL import Image
                    image_obj = Image.open(abs_image_path)
                    mm_data['image'].append(image_obj)
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            mm_data['image'].append(image)
    
    if text:
        content += text
    
    return content, mm_data

def format_vllm_input_reranker(
    query_dict: Dict[str, Any],
    doc_dict: Dict[str, Any],
    chat_template: str
):
    """
    Format query and document into vLLM input format.
    Combines multimodal data from both query and document.
    """
    query_content, query_mm_data = parse_input_dict(query_dict)
    doc_content, doc_mm_data = parse_input_dict(doc_dict)

    mm_data = { 'image': [] }
    mm_data['image'].extend(query_mm_data['image'])
    mm_data['image'].extend(doc_mm_data['image'])

    prompt = Template(chat_template).render(
        query_content=query_content,
        doc_content=doc_content,
    )
    return {
        'prompt': prompt,
        'multi_modal_data': mm_data
    }

def get_rank_scores(
    llm,
    inputs: Dict[str, Any],
    default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
    template_path: str = "reranker_template.jinja"
):
    """
    Generate relevance scores for documents given a query.
    Returns a list of scores for each document.
    """
    query_dict = inputs['query']
    doc_dicts = inputs['documents']
    instruction = inputs.get('instruction') or default_instruction

    chat_template = Template(Path(template_path).read_text())
    chat_template = chat_template.render(instruction=instruction)

    prompts = []

    for doc_dict in doc_dicts:
        prompt = format_vllm_input_reranker(
            query_dict, doc_dict, chat_template
        )
        prompts.append(prompt)

    outputs = llm.classify(
        prompts=prompts
    )
    scores = [ output.outputs.probs[0] for output in outputs ]
    return scores

