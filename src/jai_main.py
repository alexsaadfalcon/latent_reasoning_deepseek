import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from textbook_data import get_convex_latent_dataloader
from lora import apply_lora
from train import train_latent
from latent_reasoning import generate_with_latent_reasoning, generate_with_latent_reasoning_v2
from utils import format_prompt, format_answer
from experiments import get_model_predictions

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
    reasoning_steps = 30
    
    # Apply LoRA to the model
    print("Applying LoRA to the model")
    lora_dim = 8
    apply_lora(model, lora_dim=lora_dim)
    
    # Set up data
    print("Loading GSM8K dataset")
    batch_size = 4
    dataloader = get_convex_latent_dataloader(tokenizer, batch_size=batch_size, block_size=256)
    
    # Set up optimizer and scheduler
    learning_rate = 1e-3
    num_epochs = 200
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
    
    load_model = None
    # load_model = 'finetuned_latent_convex_30_0.bin'
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
            num_epochs=num_epochs,
            prefix=f'convex_{reasoning_steps}',
        )
    else:
        model.load_state_dict(torch.load(load_model))
    
    model.eval()
    
    # Test latent reasoning
    print("Testing latent reasoning")
    preds = get_model_predictions(model, tokenizer, dataloader, reasoning_steps=reasoning_steps)
    for i, pred in enumerate(preds):
        print()
        print(f'Question {i}:', pred[0])
        print()
        print(f'Correct Answer {i}:', pred[1])
        print()
        print(f'Answer {i}:', pred[2])


if __name__ == "__main__":
    main()
