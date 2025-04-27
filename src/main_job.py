import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from gsm8k import get_gsm8k_latent_dataloader
from lora import apply_lora
from train import train_latent
from utils import format_prompt, format_answer
from experiments import evaluate_accuracy

def main(learning_rate, num_epochs, reasoning_steps, lora_dim):
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
    user_id = tokenizer.encode('<｜User｜>', add_special_tokens=False)
    assistant_id = tokenizer.encode('<｜Assistant｜>', add_special_tokens=False)
    formatted_prompt_tokens = tokenizer.encode(format_prompt(''), add_special_tokens=False)
    # make sure user and assistant tokens in prompt
    assert 151644 in formatted_prompt_tokens and 151645 in formatted_prompt_tokens
    print('User and Assistant tokens', user_id, assistant_id)
    print('start/stop think ids', start_think_id, stop_think_id, newline_id)
    print('number of answer pad tokens', answer_pad)
    print('eos tokens', eos_id)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Apply LoRA to the model
    print("Applying LoRA to the model")
    apply_lora(model, lora_dim=lora_dim)

    # Set up data
    print("Loading GSM8K dataset")
    batch_size = 4
    dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128)
    
    # Set up optimizer and scheduler
    gradient_accumulation_steps = 16

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    num_update_steps = len(dataloader) // gradient_accumulation_steps * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_update_steps),
        num_training_steps=num_update_steps
    )

    print("Starting training")
    train_latent(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        reasoning_steps=reasoning_steps,
        num_epochs=num_epochs
    )
    
    model.eval()
    dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128, test=True)
    acc = evaluate_accuracy(model, tokenizer, dataloader, reasoning_steps=reasoning_steps)
    
    with open('results.json', 'w') as f:
        json.dump({'accuracy': acc}, f)

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process training parameters.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--reasoning_steps', type=int, required=True, help='Number of reasoning steps')
    parser.add_argument('--lora_dim', type=int, required=True, help='Dimension for LoRA')

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
