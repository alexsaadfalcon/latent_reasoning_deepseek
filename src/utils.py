import torch
import json

def load_json(fname):
  with open(fname, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data

def parse_answer(answer):
  # assert len(answer.split()) == 2, answer.split()
  assert answer.split()[-2] == '####'
  return answer.split()[-1]

def format_answer(answer):
  return f'\n</think>\nThe correct answer is: {answer}<｜end▁of▁sentence｜>'

def format_prompt(question):
  prompt = '<｜User｜>Solve the following math problem step-by-step. Think briefly and provide a numeric answer.\n' \
           '{}\n<｜Assistant｜><think>\n'
  return prompt.format(question)

def format_prompt_convex(question):
  prompt = 'Solve the following theoretical math problem step-by-step. Think briefly and provide a simple answer in the form of a statement or equation.\n' \
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

def collate_batch(batch, tokenizer, pad_token_id, device='cuda', block_size=128):
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
  print(inputs[0])
  print(masks[0])
  print(labels[0])
  input()
  # convert to tensor
  inputs = torch.tensor(inputs, dtype=torch.int64).to(device)
  masks = torch.tensor(masks, dtype=torch.int64).to(device)
  labels = torch.tensor(labels, dtype=torch.int64).to(device)
  return inputs, labels, masks

def collate_batch_latent(batch, tokenizer, pad_token_id, device='cuda', block_size=128):
  # For latent reasoning, we separate the inputs (questions) and answers
  # The questions will be used as input, and the answers as labels
  
  # Extract questions and answers from batch
  question_list = list(zip(*batch))[0]
  answer_list = list(zip(*batch))[1]
  
  # Tokenize questions (these will be inputs)
  questions = tokenizer(question_list, return_tensors='pt', padding=True)
  
  # Tokenize answers (these will be labels)
  answers = tokenizer(answer_list, return_tensors='pt', padding=True, add_special_tokens=False)

  return questions['input_ids'].to(device), questions['attention_mask'].to(device), \
         answers['input_ids'].to(device), answers['attention_mask'].to(device)

def prepare_latent_reasoning_batch(questions, answers, latent_steps, tokenizer, pad_token_id, device='cuda', block_size=128):
  """
  Prepares a batch for latent reasoning training.
  
  Args:
      questions: List of question strings
      answers: List of answer strings
      latent_steps: Number of latent reasoning steps to perform
      tokenizer: The tokenizer to use
      pad_token_id: The padding token ID
      device: The device to place tensors on
      block_size: Maximum sequence length
      
  Returns:
      Tuple of (input_ids, attention_mask, labels)
      Where:
      - input_ids are the tokenized questions
      - attention_mask is the mask for the inputs
      - labels are the tokenized answers
  """
  # Tokenize questions (inputs)
  question_result = tokenizer(questions, padding=True, return_tensors="pt")
  input_ids = question_result["input_ids"].to(device)
  attention_mask = question_result["attention_mask"].to(device)
  
  # Tokenize answers (labels)
  answer_result = tokenizer(answers, padding=True, return_tensors="pt")
  label_ids = answer_result["input_ids"].to(device)
  
  # Replace padding tokens with -100 in labels so they're ignored in loss computation
  label_mask = label_ids != pad_token_id
  labels = torch.where(label_mask, label_ids, -100 * torch.ones_like(label_ids))
  
  return input_ids, attention_mask, labels


