import torch
from observers.base import BaseObserver
from utils.registry import register_class

@register_class
class MinMaxObserver(BaseObserver):
    def __init__(self, symmetric=True, num_bits=8, eps=1e-8):
        self.symmetric = symmetric
        self.eps = eps
        self.min_val = 0
        self.max_val = 0
        self.num_bits = num_bits

    def observe(self, x):
        min_x = x.min().item()
        max_x = x.max().item()
        if self.min_val is None or min_x < self.min_val:
            self.min_val = min_x
        if self.max_val is None or max_x > self.max_val:
            self.max_val = max_x

    def get_scale_zero_point(self):
        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            scale = max_abs / (2 ** (self.num_bits - 1) - 1 + self.eps)
            zero_point = 0
        else:
            scale = (self.max_val - self.min_val) / (2 ** self.num_bits - 1 + self.eps)
            zero_point = round(-self.min_val / (scale + self.eps))
        return scale, zero_point 
    
    def forward(self, x):
        self.observe(x)
        scale, zero_point = self.get_scale_zero_point()
        return scale, zero_point