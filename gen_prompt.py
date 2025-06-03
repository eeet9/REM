# For generating dataset and prompt
import uuid

import pandas as pd
import torch
import random
from transformers import TextStreamer

from data_opeartion import json_to_df, save_to_jsonl, json_load
from data_fomat import jsonl_to_json
from typing import Tuple, Dict, Any
from prompt import rewrite_question_zh

"""sample from example dataset"""


def sample_from_dataset(df: pd.DataFrame, role_name: str, num_selections: int = 5) -> list[str]:
    """Randomly sample prompts for a specific role from the dataset.

    Args:
        df: Input DataFrame containing the dataset
        role_name: Role to filter by
        num_selections: Number of samples to return

    Returns:
        List of sampled prompts
    """
    role_filtered = df[df["role"] == role_name]
    sampled_prompts = role_filtered.sample(n=num_selections)["question"].tolist()
    return sampled_prompts


def generate_prompt(examples: list[str], role_name: str, template: str = rewrite_question_zh) -> str:
    """Generate a formatted prompt from example questions.

    Args:
        examples: List of example questions
        role_name: Role for context (currently unused in the template)
        template: Base template string to build upon

    Returns:
        Formatted prompt string
    """
    prompt = template

    for i, example in enumerate(examples, start=1):
        prompt += f"<task>:{example}</task>\n"

    return prompt


def do_prompt(
        model: torch.nn.Module,
        tokenizer: Any,
        datasets: pd.DataFrame,
        role_name: str,
        template: str,
        generation_config: Dict[str, Any] = None
) -> str:
    """Generate text completion using a language model with the given prompt.

    Args:
        model: The language model for text generation
        tokenizer: Tokenizer for the model
        datasets: DataFrame containing the prompt datasets
        role_name: Role name to filter prompts
        template: Template string for the prompt
        generation_config: Dictionary of generation parameters

    Returns:
        Generated text completion
    """
    # Set default generation parameters
    default_config = {
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'num_return_sequences': 1,
        'top_p': 0.9,
        'temperature': 0.7,
        'max_new_tokens': 256
    }

    # Update with user-provided config if available
    if generation_config:
        default_config.update(generation_config)

    # Generate the prompt
    n_shot_prompt = generate_prompt(datasets, role_name, template)

    # Debug output
    print("{ " * 40)
    print(f"Generated prompt:\n{n_shot_prompt}")
    print("} " * 40)

    # Prepare model inputs
    try:
        model_inputs = tokenizer(
            n_shot_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        # Set up streaming output
        streamer = TextStreamer(tokenizer)

        # Generate text
        generate_ids = model.generate(
            **model_inputs,
            streamer=streamer,
            **default_config
        )

        # Decode and clean the output
        answer = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True
        )[0]

        return answer

    except RuntimeError as e:
        print(f"Error during generation: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""


def extract_prompts(answer: str) -> list[str]:
    """Extract all prompts enclosed in <task> tags from the given string.

    Args:
        answer: Input string containing potential <task>...</task> segments

    Returns:
        List of extracted prompts (strings between <task> tags)

    Example:
        >>> extract_prompts("Some text <task>prompt1</task> more <task>prompt2</task>")
        ['prompt1', 'prompt2']
    """
    # Debug header
    debug_header = "*" * 80
    print(f"{debug_header}\nExtracting questions\n{debug_header}")

    prompts = []
    remaining_text = answer
    start_tag = "<task>"
    end_tag = "</task>"

    while True:
        # Find the next task segment
        start_idx = remaining_text.find(start_tag)
        if start_idx == -1:
            break  # No more start tags found

        end_idx = remaining_text.find(end_tag, start_idx + len(start_tag))
        if end_idx == -1:
            break  # No matching end tag found

        # Extract the prompt and update remaining text
        prompt_start = start_idx + len(start_tag)
        prompt = remaining_text[prompt_start:end_idx].strip()
        prompts.append(prompt)
        remaining_text = remaining_text[end_idx + len(end_tag):]

    # Debug output
    print(f"Extracted {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")

    return prompts


def gen_prompt(model, tokenizer, dataframe, role_name,output_path,num_prompts_to_generate):
    uniq_prompts = set([])
    new_prompts = []

    while True:
        if len(uniq_prompts) >= num_prompts_to_generate:
            break
        samples = sample_from_dataset(dataframe, role_name)
        answer = do_prompt(model, tokenizer, samples, role_name)
        prompts = extract_prompts(answer)

        for prompt in prompts:
            if prompt not in uniq_prompts:
                uniq_prompts.add(prompt)
                prompt_id = str(uuid.uuid4())
                new_prompts.append({"prompt_id": prompt_id, "prompt": prompt, "source": "generated"})
        new_prompts_df = pd.DataFrame(new_prompts)
        save_to_jsonl(new_prompts_df, new_prompts)
    return new_prompts


