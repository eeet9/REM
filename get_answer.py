import json
import uuid

import pandas as pd
import os

from openai import OpenAI
from tqdm import tqdm

from prompt import rewrite_question_en, CC_prompt_en, CU_prompt_en
from role_model import RoleModel
from data_fomat import json_load, jsonl_to_json
from data_opeartion import save_to_jsonl
from typing import List, Dict, Optional, Union, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from pathlib import Path
import time

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["DEEPSEEK_API_BASE"] = ""




def call_api_format(
        prompts: List[Dict[str, str]],
        output_file: str,
        system_prompt: str,
        api_key: str,
        base_url: str,
        model: str = "deepseek-chat",
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        min_response_length: int = 10
) -> List[Dict[str, str]]:
    """
    Process prompts through an API and save the results.

    Args:
        prompts: List of prompt dictionaries to process
        output_file: Path to save the results
        system_prompt: System prompt to use for all API calls
        api_key: API key for authentication
        base_url: Base URL for the API
        model: Model name to use (default: "deepseek-chat")
        rate_limit_delay: Delay between API calls in seconds (default: 1.0)
        max_retries: Maximum number of retries for failed API calls (default: 3)
        min_response_length: Minimum length for valid responses (default: 10)

    Returns:
        List of processed prompts with API responses
    """
    # Initialize collections and client
    unique_prompts = set()
    processed_prompts = []
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    for prompt_data in prompts:
        original_prompt = prompt_data.get("prompt", "")
        prompt_id = prompt_data.get("prompt_id", str(uuid.uuid4()))

        for attempt in range(max_retries):
            try:
                # Call API with rate limiting
                time.sleep(rate_limit_delay)

                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": original_prompt}
                    ]
                )

                response = completion.choices[0].message.content.strip()

                # Validate response
                if (len(response) >= min_response_length and
                        response not in unique_prompts):

                    unique_prompts.add(response)
                    processed_data = {
                        **prompt_data,
                        "processed_prompt": response,
                        "api_model": model,
                        "processing_timestamp": pd.Timestamp.now().isoformat()
                    }
                    processed_prompts.append(processed_data)

                    # Save periodically
                    if len(processed_prompts) % 10 == 0:
                        pd.DataFrame(processed_prompts).to_json(
                            output_file,
                            orient="records",
                            lines=True,
                            force_ascii=False
                        )

                    print(f"Processed: {response[:100]}...")  # Print first 100 chars
                    break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for prompt {prompt_id}: {str(e)}")
                if attempt == max_retries - 1:
                    processed_data = {
                        **prompt_data,
                        "processed_prompt": "",
                        "error": str(e)
                    }
                    processed_prompts.append(processed_data)
                continue

    # Final save
    pd.DataFrame(processed_prompts).to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Completed processing {len(processed_prompts)} prompts")
    return processed_prompts


def get_answer(
        model: Any,
        tokenizer: Any,
        file_path: str,
        source: str,
        output_file_path: str,
        iteration: int,
        batch_size: int = 100,
        max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate answers for prompts from a file using a language model.

    Args:
        model: Language model for text generation
        tokenizer: Tokenizer for the model
        file_path: Path to input JSON file containing prompts
        source: Source type ("CC" or others)
        output_file_path: Path to save the results
        iteration: Current iteration number (for CU answers)
        batch_size: Number of prompts to process before saving
        max_retries: Maximum number of retries for failed generations

    Returns:
        List of dictionaries containing original data and generated answers
    """
    # Load data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return []

    # Validate source
    if source not in ["CC", "CU"]:
        print(f"Warning: Unknown source type '{source}'")

    # Initialize
    new_answers = []
    system_prompt = CC_prompt_en if source == "CC" else CU_prompt_en
    answer_key = "CC_answer" if source == "CC" else f"CU_answer_round{iteration}"

    # Create output directory if needed
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Process with progress bar
    for data in tqdm(datas, desc=f"Processing {source} prompts"):
        role = RoleModel(model, tokenizer)
        prompt_text = data["prompt"] if source == "CC" else data.get("original_prompt", "")

        # Setup role model
        role.add_content(system_prompt, "answer", "system")
        role.add_content(prompt_text, "answer", "user")

        # Generate answer with retries
        response = ""
        for attempt in range(max_retries):
            try:
                response = role.run_answer()
                if response.strip():  # Only accept non-empty responses
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                response = f"ERROR: {str(e)}"

        # Store results
        result = {**data, answer_key: response}
        new_answers.append(result)

        # Periodic saving
        if len(new_answers) % batch_size == 0:
            pd.DataFrame(new_answers).to_json(
                output_file_path,
                orient="records",
                lines=True,
                force_ascii=False
            )

        # Debug output
        debug_info = {
            "iteration": iteration,
            "source": source,
            "prompt": prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
            "response": response[:100] + "..." if len(response) > 100 else response
        }
        print(json.dumps(debug_info, indent=2, ensure_ascii=False))

        role.clear_history()

    # Final save
    pd.DataFrame(new_answers).to_json(
        output_file_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Completed processing {len(new_answers)} items")
    return new_answers

