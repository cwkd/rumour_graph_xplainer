import torch
import torch.nn as nn


class LRPRelu(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPRelu, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class LRPRelu_Epsilon_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None