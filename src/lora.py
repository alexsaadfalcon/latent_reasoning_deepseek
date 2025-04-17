import math
import torch
from torch import nn

class LoRA_Linear(nn.Module):
  def __init__(self, weight, bias, lora_dim):
    super(LoRA_Linear, self).__init__()

    row, column = weight.shape

    # restore Linear
    if bias is None:
      self.linear = nn.Linear(column, row, bias=False)
      self.linear.load_state_dict({"weight": weight})
    else:
      self.linear = nn.Linear(column, row)
      # print(column, row, weight.shape, bias.shape)
      self.linear.load_state_dict({"weight": weight, "bias": bias})

    # create LoRA weights (with initialization)
    self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
    nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
    self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))
    nn.init.kaiming_uniform_(self.lora_left, a=math.sqrt(5))

  def forward(self, input):
    x = self.linear(input)
    y = input @ self.lora_right @ self.lora_left
    return x + y

def apply_lora(model):
  lora_dim = 128
  device = model.device

  # get target module name
  target_names = []
  for name, module in model.named_modules():
    if "q_proj" in name or "v_proj" in name:
      target_names.append(name)

  print('applying LoRA to', target_names)
  print(len(target_names), target_names[-1])

  # replace each module with LoRA
  for name in target_names:
    name_struct = name.split(".")
    # get target module
    module_list = [model]
    for struct in name_struct:
      module_list.append(getattr(module_list[-1], struct))

    # build LoRA
    lora = LoRA_Linear(
      weight = module_list[-1].weight,
      bias = module_list[-1].bias,
      lora_dim = lora_dim,
    ).to(device)
    # replace
    module_list[-2].__setattr__(name_struct[-1], lora)
  
  for name, param in model.named_parameters():
    if "lora_right" in name or "lora_left" in name:
      param.requires_grad = True
    else:
      param.requires_grad = False
  
  print('new model:', model)
