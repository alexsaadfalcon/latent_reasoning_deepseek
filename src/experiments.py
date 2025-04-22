import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from gsm8k import get_gsm8k_latent_dataloader
from lora import apply_lora
from train import train_latent
from latent_reasoning import generate_with_latent_reasoning_batch
from utils import format_prompt, format_answer
from tqdm import tqdm

def evaluate_accuracy(model, tokenizer, dataloader, reasoning_steps=30):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for question, question_mask, answer, answer_mask in tqdm(dataloader):
            # Move data to the device where the model is
            device = next(model.parameters()).device
            question = question.to(device)
            question_mask = question_mask.to(device)
            answer = answer.to(device)
            answer_mask = answer_mask.to(device)
            batch_size = question.size(0)
            
            # Generate predictions in batch mode
            output_tokens = generate_with_latent_reasoning_batch(
                model=model,
                tokenizer=tokenizer,
                input_ids=question,
                attention_mask=question_mask,
                reasoning_steps=reasoning_steps,
                max_new_tokens=100
            )
            
            # Process outputs and extract answers
            predictions = []
            eos_token_id = tokenizer.eos_token_id

            a_start = question.shape[1] + 9
            outputs = [tokenizer.decode(output_tokens[i, a_start:-1], skip_special_tokens=True) for i in range(batch_size)]
            eos_string = '<｜end▁of▁sentence｜>'
            predictions = [o.replace(eos_string, '').strip() for o in outputs]
            
            # Compare with ground truth answers
            for i in range(batch_size):
                # Get ground truth answer, filtering out padding tokens
                filtered_answer = answer[i].clone()
                filtered_answer = filtered_answer[filtered_answer != -100]
                true_answer = tokenizer.decode(filtered_answer, skip_special_tokens=True)
                true_answer = true_answer.split()[-1].strip()
                
                # Check if prediction matches ground truth
                if true_answer in predictions[i]:
                    total_correct += 1
            
            total_samples += batch_size
            
            # Print intermediate results
            print(f"Processed {total_samples} samples, accuracy so far: {total_correct/total_samples:.4f}")
    
    final_accuracy = total_correct / total_samples
    print(f"Final accuracy: {final_accuracy:.4f} ({total_correct}/{total_samples})")
    return final_accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_think_id = tokenizer.encode('<think>', add_special_tokens=False)
    stop_think_id = tokenizer.encode('</think>', add_special_tokens=False)
    newline_id = tokenizer.encode('\n', add_special_tokens=False)
    answer_pad = len(tokenizer.encode(format_answer(''), add_special_tokens=False))
    eos_id = tokenizer.encode('<｜end▁of▁sentence｜>', add_special_tokens=False)
    print('start/stop think ids', start_think_id, stop_think_id, newline_id)
    print('number of answer pad tokens', answer_pad)
    print('eos tokens', eos_id)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Apply LoRA to the model
    print("Applying LoRA to the model")
    lora_dim = 32
    apply_lora(model, lora_dim=lora_dim)
    
    # Set up data
    print("Loading GSM8K dataset")
    batch_size = 8
    dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128)
    
    load_model = None
    load_model = 'finetuned_latent_3.bin'
    model.load_state_dict(torch.load(load_model))
    model.eval()

    # Run accuracy evaluation
    print("Evaluating model accuracy on GSM8K dataset")
    accuracy = evaluate_accuracy(model, tokenizer, dataloader)
    print(f"Model accuracy: {accuracy:.4f}")

    print("Evaluating model accuracy on GSM8K dataset with 100 reasoning steps")
    accuracy = evaluate_accuracy(model, tokenizer, dataloader, reasoning_steps=100)
    print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
