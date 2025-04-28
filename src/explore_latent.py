import os
import torch
import pickle
import numpy as np
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
    full_tokens = []
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
            full_tokens_, attention, latent = generate_with_latent_reasoning_batch(
                model=model,
                tokenizer=tokenizer,
                input_ids=question,
                attention_mask=question_mask,
                reasoning_steps=reasoning_steps,
                max_new_tokens=100,
                temp=temp,
                output_attentions=True,
            )
            full_tokens.append(full_tokens_[0])
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

    responses = [tokenizer.decode(_full_tokens, skip_special_tokens=True) for _full_tokens in full_tokens]

    return responses, attentions, latents, q_lens, a_lens

def matching_pursuit(latents, embedding, n_nonzero=1):
    from sklearn.linear_model import OrthogonalMatchingPursuit
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
    omp.fit(embedding, latents)
    coef = omp.coef_
    return coef

def get_logits(latents, embedding):
    return embedding.T @ latents


if __name__ == '__main__':
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reasoning_steps = 30
    temp = 0.1


    dataset = 'gsm8k'
    if dataset == 'gsm8k':
        att_name = 'attention_gsm8k.pkl'
        model_fname = 'finetuned_latent_5.bin'
        dataloader = get_gsm8k_latent_dataloader(tokenizer, batch_size=batch_size, block_size=128, test=True)
        dataloader = TrimmedDataset(dataloader, 40)
    elif dataset == 'combinatorics':
        att_name = 'attention_combo.pkl'
        model_fname = 'finetuned_latent_combo_30_1.bin'
        dataloader = get_combo_latent_dataloader(tokenizer, batch_size=batch_size, block_size=256, test=False)
    else:
        raise ValueError()
    
    if not os.path.exists(att_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        if not os.path.exists('emb2.txt'):
            emb = model.get_output_embeddings().weight.detach().numpy()
            np.savetxt('emb2.txt', emb)
        else:
            emb = np.loadtxt('emb2.txt')

        lora_dim = 32
        apply_lora(model, lora_dim=lora_dim)
        model.load_state_dict(torch.load(model_fname))
        model.eval()
        responses, attentions, latents, q_lens, a_lens = get_model_attention(model, tokenizer, dataloader, reasoning_steps, temp)
        pickle.dump((responses, attentions, latents, q_lens, a_lens), open(att_name, 'wb'))
    else:
        responses, attentions, latents, q_lens, a_lens = pickle.load(open(att_name, 'rb'))
        emb = np.loadtxt('emb2.txt')
    print(attentions.shape, latents.shape)
    print(q_lens, a_lens)

    for l, latent in enumerate(latents):
        # break
        print('question', responses[l])
        for i in range(q_lens[0]-1, q_lens[0]+reasoning_steps):
            # coef = matching_pursuit(latent[i].detach().numpy(), emb.T, n_nonzero=5)
            coef = get_logits(latent[i].detach().numpy(), emb.T)
            # compute the top tokens according to coef
            # Get indices of top 5 nonzero coefficients by magnitude
            top_indices = np.abs(coef).argsort()[-10:][::-1]
            top_tokens = [tokenizer.decode([j]) for j in top_indices]
            print('top tokens:', top_tokens)
            # plt.figure()
            # plt.suptitle(f'{top_tokens}')
            # plt.stem(coef)
            # plt.show()
    print('done latent logits')

    for i in range(5):
        attention_ave = torch.mean(attentions[i:i+1], dim=(0, 1, 2)).log10()
        re_start = q_lens[i]
        re_end = re_start + reasoning_steps
        attention_ave_trim = attention_ave#[re_start:re_end, re_end:]
        plt.figure()
        plt.suptitle(f'{re_start} : {re_end}')
        plt.imshow(attention_ave_trim, aspect='auto')
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
    
