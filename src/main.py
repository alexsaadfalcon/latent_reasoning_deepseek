import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from gsm8k import get_gsm8k_latent_dataloader
from lora import apply_lora
from train import train_latent
from latent_reasoning import generate_with_latent_reasoning, generate_with_latent_reasoning_v2
from utils import format_prompt, format_answer
from experiments import evaluate_accuracy

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
    lora_dim = 64
    apply_lora(model, lora_dim=lora_dim)

    # Set up data
    print("Loading GSM8K dataset")
    batch_size = 4
    dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128)
    
    # Set up optimizer and scheduler
    learning_rate = 1e-5
    num_epochs = 10
    gradient_accumulation_steps = 16
    reasoning_steps = 10

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    num_update_steps = len(dataloader) // gradient_accumulation_steps * num_epochs
    print('number of update steps', num_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_update_steps),
        num_training_steps=num_update_steps
    )
    
    load_model = None
    # load_model = 'finetuned_latent_0.bin'
    if load_model is None:
        # Train the model
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
    else:
        model.load_state_dict(torch.load(load_model))
    
    model.eval()
    
    # Test latent reasoning
    print("Testing latent reasoning")
    test_examples = [
        format_prompt("If John has 5 apples and Mary has 3 apples, how many apples do they have in total?"),
        format_prompt("A train travels at 60 mph. How far will it travel in 3.5 hours?"),
        format_prompt("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
    ]

    for example in test_examples:
        result = generate_with_latent_reasoning_v2(
            model=model,
            tokenizer=tokenizer,
            prompt=example,
            reasoning_steps=reasoning_steps,
            max_new_tokens=100
        )
        print(f"\nInput: {example}\nOutput: {result}\n")
    
    dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128, test=True)
    acc = evaluate_accuracy(model, tokenizer, dataloader, reasoning_steps=reasoning_steps)
    print('final accuracy', acc)

if __name__ == "__main__":
    main()
