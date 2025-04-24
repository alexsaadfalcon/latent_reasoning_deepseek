from torch.utils.data import DataLoader

from utils import *

def get_convex(test=False):
  dataset = load_json('datasets/convex.json')
  dataset_ = []
  for entry in dataset:
    if entry['type'] == 'answer-based':
      dataset_.append(entry)
  dataset = dataset_
  dataset_ = []
  for entry in dataset:
    qa = dict(question=entry['text'], answer=entry['answer'])
    dataset_.append(qa)
  dataset = dataset_
  if not test:
    dataset = dataset[:21]
  else:
    dataset = dataset[21:]
  # print(dataset[0])
  return dataset

def format_convex_example(example):
  question = example['question']
  answer = example['answer']
  # Format the prompt
  context = format_prompt(question)
  # Parse the answer to get the final number
  completion = format_answer(answer)
  return context, completion

def get_convex_latent_dataloader(tokenizer, batch_size=4, block_size=128, test=False):
  """Create a dataloader specifically for latent reasoning training"""
  pad_token_id = tokenizer.pad_token_id
  
  def _collate_batch_latent(batch):
    return collate_batch_latent(batch, tokenizer, pad_token_id, block_size=block_size)
  
  train_dataset = get_convex(test=test)
  train_data = [format_convex_example(example) for example in train_dataset]
  
  dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=_collate_batch_latent
  )
  
  return dataloader
