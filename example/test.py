import torch 
import torch
from torch import nn, optim
from torch.nn import functional as F
embedding = nn.Embedding(8, 3)
input = torch.tensor([[1, 2, 1, 0], [5, 6, 7, 4]], dtype=torch.long)
output = embedding(input)

print(output,output.shape)