import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from gsm8k import get_gsm8k_dataloader
from lora import apply_lora
from train import train_latent
from latent_reasoning import generate_with_latent_reasoning

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Apply LoRA to the model
    print("Applying LoRA to the model")
    lora_dim = 16
    apply_lora(model, lora_dim=lora_dim)
    
    # Set up data
    print("Loading GSM8K dataset")
    batch_size = 2
    dataloader = get_gsm8k_dataloader(tokenizer, batch_size=batch_size)
    
    # Set up optimizer and scheduler
    learning_rate = 5e-5
    num_epochs = 2
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
    
    # Train the model
    print("Starting training")
    train_latent(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs
    )
    
    # Test latent reasoning
    print("Testing latent reasoning")
    test_examples = [
        "Solve the following math problem step-by-step. Think briefly and provide a numeric answer.\nIf John has 5 apples and Mary has 3 apples, how many apples do they have in total?",
        "Solve the following math problem step-by-step. Think briefly and provide a numeric answer.\nA train travels at 60 mph. How far will it travel in 3.5 hours?"
    ]
    
    for example in test_examples:
        result = generate_with_latent_reasoning(
            model=model,
            tokenizer=tokenizer,
            prompt=example,
            reasoning_steps=5,
            max_new_tokens=100
        )
        print(f"\nInput: {example}\nOutput: {result}\n")

if __name__ == "__main__":
    main()
