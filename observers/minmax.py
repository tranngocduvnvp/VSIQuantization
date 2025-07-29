import torch
from observers.base import BaseObserver
from utils.registry import register_class


@register_class
class MinMaxObserver(BaseObserver):
    """
    Min-max observer for quantization parameter calculation.
    
    This observer tracks the minimum and maximum values seen in the input tensors
    and uses them to compute appropriate quantization scale and zero point.
    Supports both symmetric and asymmetric quantization.
    
    Args:
        symmetric (bool): If True, use symmetric quantization around zero.
                        If False, use asymmetric quantization.
        num_bits (int): Number of bits to use for quantization
        eps (float): Small value to avoid division by zero
        
    Attributes:
        min_val (float): Minimum value observed
        max_val (float): Maximum value observed
    """
    def __init__(self, symmetric=True, num_bits=8, eps=1e-8):
        self.symmetric = symmetric
        self.eps = eps
        self.min_val = 0
        self.max_val = 0
        self.num_bits = num_bits

    def observe(self, x):
        """
        Update min/max statistics from input tensor.
        
        Args:
            x (Tensor): Input tensor to observe
            
        Updates min_val and max_val with the minimum and maximum values
        seen in the input tensor if they exceed previous extremes.
        """
        min_x = x.min().item()
        max_x = x.max().item()
        if self.min_val is None or min_x < self.min_val:
            self.min_val = min_x
        if self.max_val is None or max_x > self.max_val:
            self.max_val = max_x

    def get_scale_zero_point(self):
        """
        Compute quantization parameters based on observed min/max values.
        
        For symmetric quantization:
            - Uses the maximum absolute value to determine scale
            - Zero point is fixed at 0
            - Range is [-2^(bits-1), 2^(bits-1)-1]
            
        For asymmetric quantization:
            - Uses full min/max range to determine scale
            - Zero point shifts the range to [0, 2^bits-1]
            
        Returns:
            tuple: (scale, zero_point)
                scale: Factor to multiply quantized values by to get real values
                zero_point: Value that represents real value 0 in quantized space
        """
        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            scale = max_abs / (2 ** (self.num_bits - 1) - 1 + self.eps)
            zero_point = 0
        else:
            scale = (self.max_val - self.min_val) / (2 ** self.num_bits - 1 + self.eps)
            zero_point = round(-self.min_val / (scale + self.eps))
        return scale, zero_point 
    
    def forward(self, x):
        """
        Forward pass that both observes input and returns quantization params.
        
        Args:
            x (Tensor): Input tensor to observe
            
        Returns:
            tuple: (scale, zero_point) quantization parameters
        """
        self.observe(x)
        scale, zero_point = self.get_scale_zero_point()
        return scale, zero_point