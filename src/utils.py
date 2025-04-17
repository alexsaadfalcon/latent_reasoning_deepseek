
import torch

def parse_answer(answer):
  # assert len(answer.split()) == 2, answer.split()
  assert answer.split()[-2] == '####'
  return answer.split()[-1]

def format_prompt(question):
  prompt = 'Solve the following math problem step-by-step. Think briefly and provide a numeric answer.\n' \
           '{}\n<think>\n'
  return prompt.format(question)

def fill_ignore_label(l, c):
  l[:len(c) - 1] = [-100] * (len(c) - 1)
  return l

def pad_tokens(tokens, max_seq_length, padding_token):
  res_tokens = tokens[:max_seq_length]
  token_len = len(res_tokens)
  res_tokens = res_tokens + \
      [padding_token for _ in range(max_seq_length - token_len)]
  return res_tokens

def collate_batch(batch, tokenizer, pad_token_id, device='cuda', block_size=512):
  # tokenize both context and completion respectively
  # (context and completion is delimited by "\n")
  context_list = list(zip(*batch))[0]
  context_list = [c + "\n" for c in context_list]
  completion_list = list(zip(*batch))[1]
  context_result = tokenizer(context_list)
  context_tokens = context_result["input_ids"]
  context_masks = context_result["attention_mask"]
  completion_result = tokenizer(completion_list)
  completion_tokens = completion_result["input_ids"]
  completion_masks = completion_result["attention_mask"]
  # concatenate token
  inputs = [i + j for i, j in zip(context_tokens, completion_tokens)]
  masks = [i + j for i, j in zip(context_masks, completion_masks)]
  # create label
  eos_id = tokenizer.encode(tokenizer.eos_token)[0]
  labels = [t[1:] + [eos_id] for t in inputs]
  labels = list(map(fill_ignore_label, labels, context_tokens))
  # truncate and pad tokens
  inputs = [pad_tokens(t, block_size, pad_token_id) for t in inputs]
  masks = [pad_tokens(t, block_size, pad_token_id) for t in masks]
  labels = [pad_tokens(t, block_size, -100) for t in labels]
  # convert to tensor
  inputs = torch.tensor(inputs, dtype=torch.int64).to(device)
  masks = torch.tensor(masks, dtype=torch.int64).to(device)
  labels = torch.tensor(labels, dtype=torch.int64).to(device)
  return inputs, labels, masks

