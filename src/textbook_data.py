from torch.utils.data import DataLoader

from utils import *

def get_convex():
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

def get_convex_dataloader(tokenizer, batch_size=4, block_size=128):
  pad_token_id = tokenizer.pad_token_id
  def _collate_batch(batch):
    return collate_batch(batch, tokenizer, pad_token_id, block_size=block_size)

  train_dataset = get_convex()
  train_data = [format_convex_example(example) for example in train_dataset]
  dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=_collate_batch
  )
  return dataloader
