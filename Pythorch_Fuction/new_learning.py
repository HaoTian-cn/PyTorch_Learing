import torch
from torch import nn

m = nn.LogSigmoid()
n = nn.Sigmoid()
input = torch.tensor([1.0,-1.0,2.0,-2.0,3.0,-3.0],)
output = m(input)
output1 = n(input)
print(output1)



print(output)