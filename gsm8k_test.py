import re
import os
import json
import pickle
import requests
import numpy as np
import torch

from tqdm import tqdm
from unsloth import FastLanguageModel

# Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)

# Prepare model for inference
FastLanguageModel.for_inference(model)

# Download GSM8K test dataset
gsm_cached = 'gsm_cached.pkl'
if not os.path.exists(gsm_cached):
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    response = requests.get(url)
    resp_text = response.text
    pickle.dump(resp_text, open(gsm_cached, 'wb'))
else:
    resp_text = pickle.load(open(gsm_cached, 'rb'))

test_data = [json.loads(line) for line in resp_text.strip().split('\n')]

# Limit to n_samples
n_samples = 100
test_data = test_data[:n_samples]

# Function to extract answer from model output
def extract_answer(text):
    # Look for the answer in \boxed{} format
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for the final answer after #### or at the end of the text
    match = re.search(r'####\s*(\-?\d+\.?\d*)', text)
    if match:
        return match.group(1).strip()
    
    # If no #### format, try to find the last number in the text
    numbers = re.findall(r'(\-?\d+\.?\d*)', text)
    if numbers:
        return numbers[-1].strip()
    
    return None

# Function to extract answer from ground truth
def extract_ground_truth(answer):
    match = re.search(r'####\s*(\-?\d+\.?\d*)', answer)
    if match:
        return match.group(1).strip()
    return None


# print(test_data[0])
# print(extract_ground_truth(test_data[0]['answer']))
# exit()

# Create prompt template
prompt_template = """
Solve the following math problem step-by-step and provide a purely numeric answer:

Problem: {question}

<think>
"""
# TODO: add thinking section for COCONUT

correct_template = '\\[\n\\boxed{{{}}}\n\\]'

# Evaluation loop
correct = 0
total = 0
results = []

for sample in tqdm(test_data):
    question = sample["question"]
    ground_truth = extract_ground_truth(sample["answer"])
    
    if not ground_truth:
        continue
    
    # Prepare input
    prompt = prompt_template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
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

    # print()
    # print('question', question)
    # print()
    # print('model answer')
    # print(response)
    # print(predicted_answer)
    # print('correct', is_correct)
    # input()
    
    # Store result
    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted_answer,
        "is_correct": is_correct,
        "full_response": response
    })
    
    # Print running accuracy
    if total % 10 == 0:
        print(f"Running accuracy: {correct/total:.4f} ({correct}/{total})")

# Calculate final accuracy
final_accuracy = correct / total if total > 0 else 0
print(f"Final accuracy: {final_accuracy:.4f} ({correct}/{total})")
print('total evaluated', total)

# Save results
with open("gsm8k_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
