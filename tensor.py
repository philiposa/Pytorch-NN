import torch
import numpy as np

torch_a = torch.tensor([1,2,3,4])
torch_b = torch.tensor([5,6,7,8])
torch_a+=torch_b
print(torch_a)