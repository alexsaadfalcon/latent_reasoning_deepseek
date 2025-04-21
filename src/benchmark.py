import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from latent_reasoning import generate_with_latent_reasoning, generate_with_latent_reasoning_v2
from utils import format_prompt


def extract_answer(text):
    """Extract the answer from the model response."""
    # Look for the answer in format: "The correct answer is: X"
    answer_match = re.search(r'The correct answer is:?\s*(\-?\d+\.?\d*)', text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for the answer after ####
    match = re.search(r'####\s*(\-?\d+\.?\d*)', text)
    if match:
        return match.group(1).strip()
    
    # Look for the answer in \boxed{} format
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no structured format, try to find the last number in the text
    numbers = re.findall(r'(\-?\d+\.?\d*)', text)
    if numbers:
        return numbers[-1].strip()
    
    return None


def extract_ground_truth(answer):
    """Extract the ground truth answer from the dataset format."""
    match = re.search(r'####\s*(\-?\d+\.?\d*)', answer)
    if match:
        return match.group(1).strip()
    return None


def benchmark_standard_generation(model_name, dataset_split='test', num_samples=100,
                                max_new_tokens=200, output_file=None, device='cuda'):
    """
    Benchmark standard model generation (no latent reasoning) on GSM8K.
    
    Args:
        model_name: HuggingFace model ID or path to local model
        dataset_split: Which split of GSM8K to use ('train' or 'test')
        num_samples: Number of samples to evaluate (None for all)
        max_new_tokens: Maximum number of new tokens to generate
        output_file: Path to save detailed results JSON (None for no saving)
        device: Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    model.to(device)
    model.eval()
    
    # Load GSM8K dataset
    print(f"Loading GSM8K {dataset_split} dataset...")
    dataset = load_dataset("openai/gsm8k", 'main')
    eval_dataset = dataset[dataset_split]
    
    # Limit number of samples if specified
    if num_samples is not None:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    # Track metrics
    correct = 0
    total = 0
    results = []
    
    # Evaluation loop
    print(f"Evaluating on {len(eval_dataset)} examples with standard generation...")
    for sample in tqdm(eval_dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])
        
        if not ground_truth:
            continue
        
        # Prepare prompt
        prompt = format_prompt(question).replace("<think>\n", "")  # Remove thinking tag for standard generation
        
        # Standard generation
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted answer
        predicted_answer = extract_answer(response)
        
        # Check if correct
        is_correct = False
        if predicted_answer and ground_truth:
            # Try to normalize both answers to handle formatting differences
            try:
                pred_float = float(predicted_answer)
                gt_float = float(ground_truth)
                is_correct = abs(pred_float - gt_float) < 1e-6
            except:
                is_correct = predicted_answer.strip() == ground_truth.strip()
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store detailed result
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "is_correct": is_correct,
            "full_response": response
        })
        
        # Print running accuracy every 10 samples
        if total % 10 == 0:
            print(f"Running accuracy: {correct/total:.4f} ({correct}/{total})")
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    # Print final results
    print(f"Final accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results if requested
    if output_file:
        # Add suffix to output file
        if output_file.endswith('.json'):
            base = output_file[:-5]
            output_file = f"{base}_standard.json"
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def benchmark_latent_reasoning(model_name, dataset_split='test', num_samples=100, reasoning_steps=5, 
                               max_new_tokens=200, output_file=None, device='cuda', version='v1'):
    """
    Benchmark a latent reasoning model on GSM8K dataset.
    
    Args:
        model_name: HuggingFace model ID or path to local model
        dataset_split: Which split of GSM8K to use ('train' or 'test')
        num_samples: Number of samples to evaluate (None for all)
        reasoning_steps: Number of latent reasoning steps to use
        max_new_tokens: Maximum number of new tokens to generate
        output_file: Path to save detailed results JSON (None for no saving)
        device: Device to run evaluation on ('cuda' or 'cpu')
        version: Which version of latent reasoning to use ('v1' or 'v2')
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    model.to(device)
    model.eval()
    
    # Load GSM8K dataset
    print(f"Loading GSM8K {dataset_split} dataset...")
    dataset = load_dataset("openai/gsm8k", 'main')
    eval_dataset = dataset[dataset_split]
    
    # Limit number of samples if specified
    if num_samples is not None:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    # Track metrics
    correct = 0
    total = 0
    results = []
    
    # Choose the generation function based on version
    if version == 'v1':
        generate_func = generate_with_latent_reasoning
        print("Using original latent reasoning method (v1)")
    else:
        generate_func = generate_with_latent_reasoning_v2
        print("Using training-consistent latent reasoning method (v2)")
    
    # Evaluation loop
    print(f"Evaluating on {len(eval_dataset)} examples...")
    for sample in tqdm(eval_dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])
        
        if not ground_truth:
            continue
        
        # Prepare prompt
        prompt = format_prompt(question)
        
        # Generate response using latent reasoning
        response = generate_func(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            reasoning_steps=reasoning_steps,
            max_new_tokens=max_new_tokens
        )
        
        # Extract predicted answer
        predicted_answer = extract_answer(response)
        
        # Check if correct
        is_correct = False
        if predicted_answer and ground_truth:
            # Try to normalize both answers to handle formatting differences
            try:
                pred_float = float(predicted_answer)
                gt_float = float(ground_truth)
                is_correct = abs(pred_float - gt_float) < 1e-6
            except:
                is_correct = predicted_answer.strip() == ground_truth.strip()
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store detailed result
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "is_correct": is_correct,
            "full_response": response
        })
        
        # Print running accuracy every 10 samples
        if total % 10 == 0:
            print(f"Running accuracy: {correct/total:.4f} ({correct}/{total})")
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    # Print final results
    print(f"Final accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results if requested
    if output_file:
        # Add version suffix to output file
        if output_file.endswith('.json'):
            base = output_file[:-5]
            output_file = f"{base}_{version}.json"
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def compare_methods(model_name, dataset_split='test', num_samples=100, reasoning_steps=5,
                   max_new_tokens=200, output_dir="results", device='cuda'):
    """
    Compare different reasoning methods on the same model and dataset.
    
    Args:
        model_name: HuggingFace model ID or path to local model
        dataset_split: Which split of GSM8K to use ('train' or 'test')
        num_samples: Number of samples to evaluate (None for all)
        reasoning_steps: Number of latent reasoning steps to use
        max_new_tokens: Maximum number of new tokens to generate
        output_dir: Directory to save detailed results
        device: Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with results from all methods
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"gsm8k_{os.path.basename(model_name)}.json")
    
    # Run standard generation (baseline)
    print("\n=== Evaluating Standard Generation (Baseline) ===\n")
    standard_results = benchmark_standard_generation(
        model_name=model_name,
        dataset_split=dataset_split,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        output_file=output_file,
        device=device
    )
    
    # Run latent reasoning v1
    print("\n=== Evaluating Latent Reasoning (v1) ===\n")
    latent_v1_results = benchmark_latent_reasoning(
        model_name=model_name,
        dataset_split=dataset_split,
        num_samples=num_samples,
        reasoning_steps=reasoning_steps,
        max_new_tokens=max_new_tokens,
        output_file=output_file,
        device=device,
        version='v1'
    )
    
    # Run latent reasoning v2
    print("\n=== Evaluating Latent Reasoning (v2) ===\n")
    latent_v2_results = benchmark_latent_reasoning(
        model_name=model_name,
        dataset_split=dataset_split,
        num_samples=num_samples,
        reasoning_steps=reasoning_steps,
        max_new_tokens=max_new_tokens,
        output_file=output_file,
        device=device,
        version='v2'
    )
    
    # Summarize results
    print("\n=== Results Summary ===\n")
    print(f"Model: {model_name}")
    print(f"Dataset: GSM8K {dataset_split}")
    print(f"Number of samples: {num_samples}")
    print(f"Reasoning steps: {reasoning_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print("\nAccuracy Comparison:")
    print(f"Standard Generation: {standard_results['accuracy']:.4f} ({standard_results['correct']}/{standard_results['total']})")
    print(f"Latent Reasoning v1: {latent_v1_results['accuracy']:.4f} ({latent_v1_results['correct']}/{latent_v1_results['total']})")
    print(f"Latent Reasoning v2: {latent_v2_results['accuracy']:.4f} ({latent_v2_results['correct']}/{latent_v2_results['total']})")
    
    # Write summary to file
    summary_file = os.path.join(output_dir, f"summary_{os.path.basename(model_name)}.json")
    summary = {
        "model": model_name,
        "dataset": f"GSM8K {dataset_split}",
        "num_samples": num_samples,
        "reasoning_steps": reasoning_steps,
        "max_new_tokens": max_new_tokens,
        "standard_generation": {
            "accuracy": standard_results['accuracy'],
            "correct": standard_results['correct'],
            "total": standard_results['total']
        },
        "latent_reasoning_v1": {
            "accuracy": latent_v1_results['accuracy'],
            "correct": latent_v1_results['correct'],
            "total": latent_v1_results['total']
        },
        "latent_reasoning_v2": {
            "accuracy": latent_v2_results['accuracy'],
            "correct": latent_v2_results['correct'],
            "total": latent_v2_results['total']
        }
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a latent reasoning model on GSM8K.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="HuggingFace model ID or path to local model")
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "test"],
                        help="Which split of GSM8K to use")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--reasoning_steps", type=int, default=5,
                        help="Number of latent reasoning steps to use")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save detailed results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run evaluation on")
    parser.add_argument("--method", type=str, default="compare", 
                        choices=["standard", "latent_v1", "latent_v2", "compare"],
                        help="Which method to evaluate (or compare all)")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"],
                        help="Version of latent reasoning to use (v1=original, v2=training-consistent)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"gsm8k_{os.path.basename(args.model_name)}.json")
    
    # Run the specified method(s)
    if args.method == "compare":
        compare_methods(
            model_name=args.model_name,
            dataset_split=args.dataset_split,
            num_samples=args.num_samples,
            reasoning_steps=args.reasoning_steps,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            device=args.device
        )
    elif args.method == "standard":
        benchmark_standard_generation(
            model_name=args.model_name,
            dataset_split=args.dataset_split,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            output_file=output_file,
            device=args.device
        )
    elif args.method == "latent_v1":
        benchmark_latent_reasoning(
            model_name=args.model_name,
            dataset_split=args.dataset_split,
            num_samples=args.num_samples,
            reasoning_steps=args.reasoning_steps,
            max_new_tokens=args.max_new_tokens,
            output_file=output_file,
            device=args.device,
            version="v1"
        )
    elif args.method == "latent_v2":
        benchmark_latent_reasoning(
            model_name=args.model_name,
            dataset_split=args.dataset_split,
            num_samples=args.num_samples,
            reasoning_steps=args.reasoning_steps,
            max_new_tokens=args.max_new_tokens,
            output_file=output_file,
            device=args.device,
            version="v2"
        )
