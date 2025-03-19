import torch
from torch import nn
# retireved from llama 3 source code 
# https://github.com/meta-llama/llama3/blob/main/llama/model.py
# We use RMSNorm more currently due to its computational efficiency and stability
# - RMSNorm skips mean subtraction, only computing the RMS value, which is much cheaper.
# - Transformers are sensitive to LayerNorm because the mean subtraction step introduces zero-centered activations, which can cause instabilities in deep models.
# - RMSNorm treats each feature independently and only rescales values, allowing more flexibility for learning representations.
# - RMSNorm does not depend on batch statistics, making it a better choice for small-batch inference and fine-tuning.

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
# layer normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
# math: y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
# equation adopted from pytorch: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/normalization.py
# below is a simplified implementation. Gamma is the weight and beta is the bias.
# Benefits:
# Independent of batch size
# Stabilizes training in deep networks
# Better for recurrent models (RNNs, Transformers)
# Faster convergence
# More effective for non-IID data
# Element-wise affine transformation
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # normalization over the last dimension, consider the feature dimension of an input sequence in a batch
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        return output * self.weight + self.bias # affine transformation

# batch normalization: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
# we consider mainly for NLP so look at batchnorm 1d is fine
# Batch norm computes the mean and std over the batch, hidden size dim. Think of a 3d cube.
# math: y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
# Benefits of Batch Normalization
# - Reduces internal covariate shift, stabilizing training
# - Enables higher learning rates, improving convergence speed
# - Improves gradient flow by normalizing activations
# - Acts as a regularizer, reducing overfitting similar to dropout
# - Reduces dependence on weight initialization, making training more robust
# - Improves training stability, preventing divergence
class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):
        if self.training:
            # compute mean and variance over batch and spatial dimensions
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
            # update running statistics for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze() # squeeze to fit the running_mean size
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze() # squeeze to fit the running_var size
        else:
            # use stored running mean and variance for inference
            mean = self.running_mean.view(1, -1, 1)
            var = self.running_var.view(1, -1, 1)
        # normalize the input
        output = (x - mean) / torch.sqrt(var + self.eps)
        return output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1) # affine transformation

# Instance Normalization: https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html
# Note: this is not frequently used in NLP applications, it looks similar to layernorm
# but in the field of CNN, it is applied to channels and thus dim=-1.
# Benefits of Instance Normalization
# - Works well for style transfer and tasks where batch statistics are unstable
# - Normalizes each instance independently, making it batch size independent
# - Effective for generative models like GANs where batch statistics vary
# - Useful in online learning and reinforcement learning where batch sizes are small
class InstanceNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Compute mean and variance per sample independently
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # Normalize input per sample
        output = (x - mean) / torch.sqrt(var + self.eps)
        return output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

# Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
# Benefits of Group Normalization
# - Works well for small batch sizes where BatchNorm fails
# - Reduces dependency on batch statistics, making it stable across different batch sizes
# - Groups channels for normalization, allowing better flexibility than InstanceNorm
# - Often used in computer vision tasks like object detection and segmentation
class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        N, C, *dims = x.shape # unpack all other dimensions
        # Reshape input into groups
        x = x.view(N, self.num_groups, C // self.num_groups, *dims) # this is expanding a dim
        # Compute mean and variance within each group
        mean = x.mean(dim=(2, *range(3, x.dim())), keepdim=True)
        var = x.var(dim=(2, *range(3, x.dim())), unbiased=False, keepdim=True)
        # Normalize within each group
        output = (x - mean) / torch.sqrt(var + self.eps)
        # reshape to input.shape
        output = output.view(N, C, *dims)
        # Ensure weight and bias are correctly broadcasted across spatial dimensions
        # self.weight.view(1, C, *([1] * (x.dim() - 2))) expands weight across all spatial dimensions
        # self.bias.view(1, C, *([1] * (x.dim() - 2))) expands bias similarly
        # again, use * to unpack the list to a list with 1 and len x.dim - 2 for all other *dim.
        return output * self.weight.view(1, C, *([1] * (x.dim() - 2))) + self.bias.view(1, C, *([1] * (x.dim() - 2)))

# Built-in BatchNorm, InstanceNorm, and GroupNorm for comparison
batch_norm = nn.BatchNorm1d(num_features=4, eps=1e-6, momentum=0.1, affine=True)
instance_norm = nn.InstanceNorm1d(num_features=4, eps=1e-6, affine=True)
group_norm = nn.GroupNorm(num_groups=2, num_channels=4, eps=1e-6, affine=True)

# Input Tensor (2, 4)
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 0.1, 0.2, 0.3]
])

# Apply LayerNorm
layer_norm = LayerNorm(dim=x.shape[-1])
out_x_ln = layer_norm(x)
print("LayerNorm Output:\n", out_x_ln, "\nshape:", out_x_ln.shape)

# Apply BatchNorm (requires 3D input with batch dimension first)
x_bn = x.unsqueeze(0).repeat(2, 1, 1)  # Simulating batch dimension
custom_batch_norm = BatchNorm(num_features=4)
out_x_bn_custom = custom_batch_norm(x_bn)
print("\nCustom BatchNorm Output:\n", out_x_bn_custom, "\nshape:", out_x_bn_custom.shape)

# Apply InstanceNorm (also expects (batch, channels, seq))
custom_instance_norm = InstanceNorm(num_features=4)
out_x_in_custom = custom_instance_norm(x_bn)
print("\nCustom InstanceNorm Output:\n", out_x_in_custom, "\nshape:", out_x_in_custom.shape)

# Apply GroupNorm
custom_group_norm = GroupNorm(num_groups=2, num_features=4)
out_x_gn_custom = custom_group_norm(x_bn.unsqueeze(1))  # Adding extra dim for group norm compatibility
print("\nCustom GroupNorm Output:\n", out_x_gn_custom.squeeze(1), "\nshape:", out_x_gn_custom.shape)