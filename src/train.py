import os
import math
import torch
from torch.nn import functional as F


def train_latent(model, optimizer, scheduler, dataloader, batch_size=4,
                 gradient_accumulation_steps=16, num_epochs=2):
  if os.path.exists("loss.txt"):
    os.remove("loss.txt")

  num_update_steps = math.ceil(len(dataloader) / batch_size / gradient_accumulation_steps)
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    model.train()
    for i, (inputs, labels, masks) in enumerate(dataloader):
      with torch.set_grad_enabled(True):
        outputs = model(
            input_ids=inputs,
            attention_mask=masks,
        )
        loss = F.cross_entropy(outputs.logits.transpose(1,2), labels)
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
  torch.save(model.state_dict(), "finetuned_latent.bin")


if __name__ == '__main__':
  pass
