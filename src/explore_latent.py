import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from gsm8k import get_gsm8k_latent_dataloader
from textbook_data import get_combo_latent_dataloader

from lora import apply_lora
from latent_reasoning import generate_with_latent_reasoning_batch
from utils import *

def get_model_attention(model, tokenizer, dataloader, reasoning_steps=30, temp=0.0):
    model.eval()
    attentions = []

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
            attention = generate_with_latent_reasoning_batch(
                model=model,
                tokenizer=tokenizer,
                input_ids=question,
                attention_mask=question_mask,
                reasoning_steps=reasoning_steps,
                max_new_tokens=100,
                temp=temp,
                output_attentions=True,
            )
            attention = torch.stack(attention, dim=1)
            attentions.append(attention)
    
    attentions_ = []
    # zero pad all attentions to have the same shape in dimension 2, sequence length
    for att in attentions:
        att_pad = torch.zeros_like(attentions[-1])
        seq_len = att.shape[2]
        att_pad[:, :, :seq_len, :, :] = att
        attentions_.append(att_pad)
    attentions = attentions_

    return torch.cat(attentions, dim=0)


if __name__ == '__main__':
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    lora_dim = 32
    apply_lora(model, lora_dim=lora_dim)

    reasoning_steps = 30
    temp = 0.1

    dataloader = get_combo_latent_dataloader(tokenizer, batch_size=batch_size, block_size=256)
    attentions = get_model_attention(model, tokenizer, dataloader, reasoning_steps, temp)
    print(attentions.shape)
    exit()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Attention Heads Visualization {input_ids.shape[1]} Question Tokens')
    
    # Plot 4 attention heads in a 2x2 grid
    for row in range(2):
        for col in range(2):
            head_idx = row * 2 + col
            ax = axs[row, col]
            attention_matrix = last_layer_attention[0, head_idx].detach().log10().cpu().numpy()
            im = ax.imshow(attention_matrix, cmap='viridis')
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key tokens')
            ax.set_ylabel('Query tokens')
    
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.tight_layout()
    plt.savefig(f'attention_matrix_step_{i}.png')
    # plt.close()
    plt.show()
    
