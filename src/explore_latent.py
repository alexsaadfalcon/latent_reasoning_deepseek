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


    dataset = 'gsm8k'
    if dataset == 'gsm8k':
        att_name = 'attention_gsm8k.pkl'
        model_name = 'finetuned_latent_5.bin'
        dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128, test=True)
        dataloader = TrimmedDataset(dataloader, 100)
    elif dataset == 'combinatorics':
        att_name = 'attention_combo.pkl'
        model_name = 'finetuned_latent_combo_30_0.bin'
        dataloader = get_combo_latent_dataloader(tokenizer, batch_size=batch_size, block_size=256, test=True)
    else:
        raise ValueError()

    model.load_state_dict(torch.load(model_name))
    model.eval()
    
    if not os.path.exists(att_name):
        attentions, latents, q_lens, a_lens = get_model_attention(model, tokenizer, dataloader, reasoning_steps, temp)
        pickle.dump((attentions, latents, q_lens, a_lens), open(att_name, 'wb'))
    else:
        attentions, latents, q_lens, a_lens = pickle.load(open(att_name, 'rb'))
    print(attentions.shape, latents.shape)
    print(q_lens, a_lens)

    for i in range(attentions.shape[0]):
      attention_ave = torch.mean(attentions[i:i+1], dim=(0, 1, 2)).log10()
      plt.figure()
      plt.imshow(attention_ave)
      plt.colorbar()

    # show cosine alignment between latents for sample 0
    # Calculate pairwise cosine similarity for latents from first sample
    for i in range(5):
      sample_latents = latents[i]  # Shape: [num_steps, hidden_dim]
      # trim to just the latent tokens
      sample_latents = sample_latents[q_lens[0]:q_lens[0] + reasoning_steps]
      
      # Compute cosine similarity matrix
      with torch.no_grad():
        norm = torch.norm(sample_latents, dim=1, keepdim=True)
        normalized_latents = sample_latents / norm
        cosine_sim = torch.mm(normalized_latents, normalized_latents.t())
      
      # Plot the similarity matrix
      plt.figure()
      plt.imshow(cosine_sim.cpu().numpy())
      plt.colorbar()
      plt.title("Pairwise Cosine Similarity of Latents")
      plt.xlabel("Latent Step")
      plt.ylabel("Latent Step") 
    
    with torch.no_grad():
      last_inds = [q_lens[i] + reasoning_steps - 1 for i in range(len(latents))]
      last_latent = [latents[i, last_inds[i], :] for i in range(len(latents))]
      last_latent = torch.stack(last_latent, dim=0)
      norm = torch.norm(last_latent, dim=1, keepdim=True)
      norm_last_latent = last_latent / norm
      last_latent_cosine_sim = torch.mm(norm_last_latent, norm_last_latent.t())

    plt.figure()
    plt.imshow(last_latent_cosine_sim.cpu().numpy())
    plt.colorbar()
    plt.title("Pairwise Cosine Similarity of Last Latent")
    plt.xlabel("Latent Step")
    plt.ylabel("Latent Step")

    plt.show()
    
