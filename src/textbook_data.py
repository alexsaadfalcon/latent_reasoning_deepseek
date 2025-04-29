from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from utils import *


DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


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
  context = format_prompt_convex(question)
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

def get_combo(test=False):
  dataset = load_json('datasets/combinatorics.json')
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
    dataset = dataset[:8]
  else:
    dataset = dataset[8:]
  # print(dataset[0])
  return dataset

def format_combo_example(example):
  question = example['question']
  answer = example['answer']
  # Format the prompt
  context = format_prompt_combo(question)
  # Parse the answer to get the final number
  completion = format_answer(answer)
  return context, completion

def get_combo_latent_dataloader(tokenizer, batch_size=4, block_size=128, test=False):
  """Create a dataloader specifically for latent reasoning training"""
  pad_token_id = tokenizer.pad_token_id
  
  def _collate_batch_latent(batch):
    return collate_batch_latent(batch, tokenizer, pad_token_id, block_size=block_size)
  
  train_dataset = get_combo(test=test)
  train_data = [format_combo_example(example) for example in train_dataset]
  
  dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=_collate_batch_latent
  )
  
  return dataloader


class LocalPreTrainedModel:
  def __init__(self, model_name):
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    self.model.to(self.device)


class DeepSeekModel(LocalPreTrainedModel):
  def __init__(self):
    model_name = DEEPSEEK_MODEL_NAME
    super().__init__(model_name)

  def generate_response(self, prompt, **kwargs):
    model_input = "<｜User｜>" + prompt + "<｜Assistant｜>"
    model_input = self.tokenizer(model_input, return_tensors="pt").to(self.device)

    with torch.no_grad():
      model_output = self.model.generate(**model_input, **kwargs, pad_token_id=self.tokenizer.eos_token_id)

    response = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
    return response


def solve_convex_optimization_problems(chapter=2, max_new_tokens=2048, temperature=0.1):
  """
    Solve convex optimization problems using the DeepSeek model.
    The problems are taken from the book "Convex Optimization" by Boyd and Vanderberghe.
  """
  problems = json.load(open("src/datasets/problems/convex" + str(chapter) + ".json"))
  # Filter problems to only include answer-based problems
  problems = [problem for problem in problems if problem["type"] == "answer-based"]

  deepseek_model = DeepSeekModel()

  for problem in problems:
    problem_text = problem["text"]
    prompt = format_prompt_convex(problem_text)
    solution = deepseek_model.generate_response(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    filename = "data/output/Convex_Optimization_Boyd_Vanderberghe_" + str(chapter) + "_problems_solutions_" + problem["exercise"] + ".json"
    with open(filename, 'w') as f:
      json.dump({"problem": problem_text, "solution": solution}, f, indent=4)


def solve_combinatorics_problems(chapter=1, max_new_tokens=2048, temperature=0.1):
  """
    Solve combinatorics problems using the DeepSeek model.
    The problems are from the book "Aspects of Combinatorics" by Victor Bryant.
  """
  problems = json.load(open("src/datasets/problems/combinatorics" + str(chapter) + ".json"))
  # Filter problems to only include answer-based problems
  problems = [problem for problem in problems if problem["type"] == "answer-based"]

  deepseek_model = DeepSeekModel()

  for problem in problems:
    problem_text = problem["text"]
    prompt = format_prompt_convex(problem_text)
    solution = deepseek_model.generate_response(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    filename = "data/output/Aspects_of_Combinatorics_Victor_Bryant_" + str(chapter) + "_problems_solutions_" + problem["exercise"] + ".json"
    with open(filename, 'w') as f:
      json.dump({"problem": problem_text, "solution": solution}, f, indent=4)
