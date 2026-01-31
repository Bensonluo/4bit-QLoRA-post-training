"""Qualitative evaluation through text generation."""

from typing import List, Dict

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils import console


def generate_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    num_samples: int = 5,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    """Generate text samples from model.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        dataset: Dataset with prompts
        num_samples: Number of samples to generate
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of dictionaries with prompt and response
    """
    console.print(f"[cyan]Generating {num_samples} samples...[/cyan]\n")

    model.eval()
    generations = []

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]

        # Extract prompt
        if "instruction" in example:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        elif "prompt" in example:
            prompt = example["prompt"]
        elif "text" in example:
            # Take first 100 chars as prompt
            text = example["text"]
            prompt = text[:100] if len(text) > 100 else text
        else:
            continue

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):]

        generations.append({
            "prompt": prompt,
            "response": response.strip(),
        })

    console.print(f"[green]✓ Generated {len(generations)} samples[/green]")

    return generations


def interactive_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> None:
    """Interactive text generation loop.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        max_length: Maximum generation length
    """
    console.print("\n[bold cyan]=== Interactive Generation ===[/bold cyan]")
    console.print("[yellow]Type 'quit' or 'exit' to stop[/yellow]\n")

    model.eval()

    while True:
        # Get prompt
        prompt = console.input("[bold cyan]Prompt:[/bold cyan] ")

        if prompt.lower() in ["quit", "exit", "q"]:
            console.print("[yellow]Exiting...[/yellow]")
            break

        if not prompt:
            continue

        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        # Generate
        console.print("[cyan]Generating...[/cyan]")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):]

        console.print(f"\n[bold green]Response:[/bold green]")
        console.print(response)
        console.print()
