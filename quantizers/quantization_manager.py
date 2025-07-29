from torch.backends.cuda import flash_sdp_enabled
from utils.registry import CLASS_REGISTRY
from quantizers.uniform import *
from observers.minmax import *
import torch
import torch.nn as nn
import numpy as np


class QuantizationManager(nn.Module):
    """
    Manager class for handling quantization of neural network tensors.
    
    This class combines observers and quantizers to handle the complete
    quantization process, including parameter collection, scale/zero-point
    computation, and actual quantization.
    
    Args:
        quantizer_name (str): Name of quantizer class to use
        observer_name (str): Name of observer class to use
        bits_width (int): Number of bits for quantization
        is_symmetric (bool): Whether to use symmetric quantization
        is_learning_scale (bool): Whether to learn quantization parameters
        
    Attributes:
        quantizer: Instance of quantizer class
        observer: Instance of observer class
        scale: Quantization scale factor
        zero_point: Quantization zero point
        mean_x (list): History of mean absolute values
        std (list): History of standard deviations
    """
    def __init__(self, 
        quantizer_name: str,
        observer_name: str,
        bits_width: int,
        is_symmetric: bool,
        is_learning_scale: bool = True,
    ) -> None:
        super(QuantizationManager, self).__init__()
        self.quantizer = CLASS_REGISTRY[quantizer_name](bits_width, is_symmetric)
        self.observer = CLASS_REGISTRY[observer_name](is_symmetric)
        self.is_learning_scale = is_learning_scale
        self.bits_width = bits_width
        self.scale = 1
        self.zero_point = 0
        self.is_observer_qparam = True
        self.is_learning_scale = is_learning_scale
        self.is_quantize = True
        self.is_symmetric = True
        self.mean_abs_x = []
        self.mean_x = []
        self.std = []

    def collect_qparameter(self, x):
        """
        Collect quantization parameters from input tensor.
        
        If not learning scale and observer is enabled, updates scale and
        zero point based on tensor statistics.
        
        Args:
            x (Tensor): Input tensor to analyze
        """
        if not self.is_learning_scale and self.is_observer_qparam:
            self.mean_abs_x.append(torch.mean(torch.abs(x.detach())).cpu().item())
            self.mean_x.append(torch.mean(x.detach()).cpu().item())
            self.std.append(torch.std(x.detach()).cpu().item())
            scale, zero_point = self.observer.forward(x)
            self.scale = scale
            self.zero_point = zero_point
    
    def quantize(self, x):
        """
        Quantize input tensor using current parameters.
        
        First collects parameters if needed, then applies quantization
        if enabled.
        
        Args:
            x (Tensor): Input tensor to quantize
            
        Returns:
            Tensor: Quantized tensor, or original if quantization disabled
        """
        self.collect_qparameter(x)
        if self.is_quantize:
            return self.quantizer.quantize(x, self.scale, self.zero_point, self.is_learning_scale)
        else:
            return x

    def make_learn_qparameter(self):
        """
        Convert scale and zero point to learnable parameters.
        
        For symmetric quantization, zero point is fixed at 0.
        For asymmetric quantization, both parameters become learnable.
        """
        self.scale = nn.Parameter(torch.tensor(self.scale), requires_grad=True)
        if not self.is_symmetric:
            self.zero_point = nn.Parameter(torch.tensor(self.zero_point+1e-9), requires_grad=True)
        else:
            self.zero_point = 0
    
    def init_scaling_factor_for_learning(self):
        """
        Initialize scale factor for learnable quantization.
        
        Uses the mean of collected mean absolute values to set initial scale,
        adjusted for the number of quantization levels.
        """
        self.scale = 2*np.mean(self.mean_abs_x)/np.sqrt(2**(self.bits_width-1)-1)
        # self.scale = max(np.abs(np.mean(self.mean_x)-3*np.mean(self.std)),\
        #      np.abs(np.mean(self.mean_x)+3*np.mean(self.std)))/(2**(self.bits_width-1)-1)
    
    def winsorized_mean(self, x, lower=5, upper=95, sample_size=1000000):
        """
        Compute winsorized mean of tensor to reduce impact of outliers.
        
        Args:
            x (Tensor): Input tensor
            lower (float): Lower percentile for winsorization (0-100)
            upper (float): Upper percentile for winsorization (0-100)
            sample_size (int): Maximum number of values to use
            
        Returns:
            float: Winsorized mean of the tensor
            
        Note:
            If tensor has more elements than sample_size, randomly samples
            that many elements to compute percentiles and mean.
        """
        x_flat = x.flatten()
        if x_flat.numel() > sample_size:
            indices = torch.randperm(x_flat.numel())[:sample_size]
            sample = x_flat[indices]
        else:
            sample = x_flat

        lower_val = torch.quantile(sample, lower / 100)
        upper_val = torch.quantile(sample, upper / 100)

        mask = (x_flat >= lower_val) & (x_flat <= upper_val)
        trimmed_x = x_flat[mask]
        return trimmed_x.mean()