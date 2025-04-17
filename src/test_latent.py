import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from gsm8k import get_gsm8k_dataloader
from lora import apply_lora
from train import train_latent
from latent_reasoning import generate_with_latent_reasoning, latent_reasoning_forward

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
  apply_lora(model)
  
  # Test latent reasoning
  print("Testing latent reasoning")
  template = "Solve the following math problem step-by-step. " \
    "Think briefly and provide a numeric answer.\n{}\n<think>\n"
  test_examples = [
    template.format("If John has 5 apples and Mary has 3 apples, how many apples do they have in total?"),
    template.format("A train travels at 60 mph. How far will it travel in 3.5 hours?"),
  ]

  # Process examples in batch
  # batch_inputs = tokenizer(test_examples, padding=True, return_tensors="pt").to(device)
  batch_inputs = tokenizer(test_examples, padding=True, padding_side='right', return_tensors="pt").to(device)
  input_ids = batch_inputs.input_ids
  attention_mask = batch_inputs.attention_mask
  print(attention_mask)
  
  # Process the batch through latent reasoning
  print(input_ids.shape, attention_mask.shape)
  embeds, mask, types = latent_reasoning_forward(model, input_ids, attention_mask)
  print(embeds.shape, mask.shape, types.shape)

if __name__ == '__main__':
  main()
