from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import *

def get_gsm8k():
  dataset = load_dataset("openai/gsm8k", 'main')
  train_dataset = dataset["train"]
  train_dataset = train_dataset.select(range(1000))
  return train_dataset

def format_gsm8k_example(example):
  question = example['question']
  answer = example['answer']
  # Format the prompt
  context = format_prompt(question)
  # Parse the answer to get the final number
  completion = parse_answer(answer)
  return context, completion

def get_gsm8k_dataloader(tokenizer, batch_size=4, block_size=512):
  pad_token_id = tokenizer.pad_token_id
  def _collate_batch(batch):
    return collate_batch(batch, tokenizer, pad_token_id, block_size=block_size)

  train_dataset = get_gsm8k()
  train_data = [(format_gsm8k_example(example)) for example in train_dataset]
  dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=_collate_batch
  )
  return dataloader
