import os
import torch
import pickle
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
    latents = []
    q_lens, a_lens = [], []

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
            attention, latent = generate_with_latent_reasoning_batch(
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
            latents.append(latent)
            q_lens.append(question.shape[1])
            a_lens.append(answer.shape[1])
    
    attentions_ = []
    # attentions have shape (batch, layer, head, seq_len, seq_len)
    # zero pad all attentions to have the same shape in dimension 3-4
    longest_seq = max(_a.shape[4] for _a in attentions)
    n_layers = attentions[0].shape[1]
    n_heads = attentions[0].shape[2]
    att_shape = (1, n_layers, n_heads, longest_seq, longest_seq)
    for att in attentions:
        att_pad = torch.zeros(att_shape)
        seq_len = att.shape[4]
        att_pad[:, :, :, :seq_len, :seq_len] = att
        attentions_.append(att_pad)
    attentions = attentions_
    attentions = torch.cat(attentions, dim=0)
    
    latents_ = []
    # latents have shape (batch, seq_len, embedding_dim)
    # zero pad all latents to have the same shape in dimension 1
    emb_dim = latents[0].shape[2]
    lat_shape = (1, longest_seq, emb_dim)
    for lat in latents:
        lat_pad = torch.zeros(lat_shape)
        seq_len = lat.shape[1]
        lat_pad[:, :seq_len, :] = lat
        latents_.append(lat_pad)
    latents = latents_
    latents = torch.cat(latents, dim=0)

    return attentions, latents, q_lens, a_lens


if __name__ == '__main__':
    batch_size = 1
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
    if not os.path.exists('attention.pkl'):
        attentions, latents, q_lens, a_lens = get_model_attention(model, tokenizer, dataloader, reasoning_steps, temp)
        pickle.dump((attentions, latents, q_lens, a_lens), open('attention.pkl', 'wb'))
    else:
        attentions, latents, q_lens, a_lens = pickle.load(open('attention.pkl', 'rb'))
    print(attentions.shape, latents.shape)
    print(q_lens, a_lens)

    attention_ave = torch.mean(attentions, dim=(0, 1, 2)).log10()
    plt.figure()
    plt.imshow(attention_ave)
    plt.colorbar()
    plt.show()
    
