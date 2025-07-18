import torch
from quantizers.base import BaseQuantizer
from utils.registry import register_class

@register_class
class UniformQuantizer(BaseQuantizer):
    def __init__(self, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.symmetric = symmetric

    def quantize(self, x, scale, zero_point):
        qmin = 0
        qmax = 2 ** self.num_bits - 1
        if self.symmetric:
            qmin = - (2 ** (self.num_bits - 1))
            qmax = 2 ** (self.num_bits - 1) - 1
        x_int = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant 
        