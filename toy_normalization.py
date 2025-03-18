import torch
from torch import nn
# retireved from llama 3 source code 
# https://github.com/meta-llama/llama3/blob/main/llama/model.py

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # epslion to prevent devision by zero
        self.weight = nn.Parameter(torch.ones(dim)) # learnable param, only need to be applied to the feature dimension

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # x / (RMS(x) + \epsilon)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight # elemntwise product

# note Root Mean Square Normalization (RMSNorm)
# apply to a vector (2, 4):
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 0.1, 0.2, 0.3]
])
rms_x = RMSNorm(dim=x.shape[-1], eps=1e-6)
out_x = rms_x(x)
print(out_x, f"\nshape: {out_x.shape}")
# tensor([[0.3651, 0.7303, 1.0954, 1.4606],
#        [1.6013, 0.3203, 0.6405, 0.9608]], grad_fn=<MulBackward0>) shape: torch.Size([2, 4])

# apply to a batched input (4, 2, 4):
y = x.unsqueeze(0).repeat(2, 1, 1) # repeat(dim1, dim2, ...) -> tensor repeating each dim as the input times
rms_y = RMSNorm(dim=y.shape[-1], eps=1e-6)
out_y = rms_y(y)
print(out_y, f"\nshape: {out_y.shape}")
'''
tensor([[[0.3651, 0.7303, 1.0954, 1.4606],
         [1.6013, 0.3203, 0.6405, 0.9608]],

        [[0.3651, 0.7303, 1.0954, 1.4606],
         [1.6013, 0.3203, 0.6405, 0.9608]]], grad_fn=<MulBackward0>) shape: torch.Size([2, 2, 4])
'''