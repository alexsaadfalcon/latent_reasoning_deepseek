import os
import math
import torch
from torch.nn import functional as F

from latent_reasoning import latent_reasoning_forward, latent_plus_answer_loss, \
  latent_reasoning_forward_detach, latent_reasoning_forward_one_step_gradients

def train_latent(model, optimizer, scheduler, dataloader, batch_size=4,
                 gradient_accumulation_steps=16, num_epochs=2, reasoning_steps=30, prefix=None):
  if os.path.exists("loss.txt"):
    os.remove("loss.txt")

  num_update_steps = math.ceil(len(dataloader) / batch_size / gradient_accumulation_steps)
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    model.train()
    for i, (question, question_mask, answer, answer_mask) in enumerate(dataloader):
      with torch.set_grad_enabled(True):
        # Process the input through latent reasoning
        # embeds, all_masks, _ = latent_reasoning_forward(model, question, question_mask, reasoning_steps=30)
        embeds, all_masks, _ = latent_reasoning_forward_detach(model, question, question_mask, reasoning_steps=reasoning_steps)
        # embeds, all_masks, _ = latent_reasoning_forward_one_step_gradients(model, question, question_mask)
        ### CHECK THAT THIS IS FORMATTED CORRECTLY
        # print(question)
        # print(answer)
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        # print('input0', tokenizer.decode(question[0]))
        # # Filter out -100 padding tokens before decoding
        # filtered_labels = answer[0].clone()
        # filtered_labels = filtered_labels[filtered_labels != -100]
        # print('label0', tokenizer.decode(filtered_labels))
        # exit()
        loss = latent_plus_answer_loss(model, embeds, all_masks, answer, answer_mask)
        loss.backward()
        if ((i + 1) % gradient_accumulation_steps == 0) or \
            (i + 1 == len(dataloader)):
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()

          print(f"Epoch {epoch+1} "
                f"{math.ceil((i + 1) / batch_size / gradient_accumulation_steps)}"
                f"/{num_update_steps} - loss: {loss.item() :2.4f}", end="\r")

      # record loss
      with open("loss.txt", "a") as f:
        f.write(str(loss.item()))
        f.write("\n")
    print("")

  # save model
  counter = 0
  while True:
    if prefix:
      fname = f"finetuned_latent_{prefix}_{counter}.bin"
    else:
      fname = f"finetuned_latent_{counter}.bin"
    counter += 1
    if not os.path.exists(fname):
      torch.save(model.state_dict(), fname)
      break


if __name__ == '__main__':
  pass
