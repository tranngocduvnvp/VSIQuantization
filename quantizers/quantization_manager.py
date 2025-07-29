from torch.backends.cuda import flash_sdp_enabled
from utils.registry import CLASS_REGISTRY
from quantizers.uniform import *
from observers.minmax import *
import torch
import torch.nn as nn
import numpy as np


class QuantizationManager(nn.Module):
    def __init__(self, 
        quantizer_name:str,
        observer_name:str,
        bits_width:int,
        is_symmetric: bool,
        is_learning_scale:bool=True,
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
        self.is_symmetric=True
        self.mean_x = []
        self.std = []

    def collect_qparameter(self, x):
        # self.mean_x.append(self.winsorized_mean(torch.abs(x.detach())).cpu().item())
        if self.is_learning_scale == False and self.is_observer_qparam ==True:
            self.mean_x.append(torch.mean(torch.abs(x.detach())).cpu().item())
            # self.std.append(torch.mean(torch.abs(x.detach())).cpu().item())
            scale, zero_point = self.observer.forward(x)
            self.scale = scale
            self.zero_point = zero_point
        else:
            pass
    
    def quantize(self, x):
        self.collect_qparameter(x)
        if self.is_quantize == True:
            return self.quantizer.quantize(x, self.scale, self.zero_point, self.is_learning_scale)
        else:
            return x

    def make_learn_qparameter(self):
        self.scale = nn.Parameter(torch.tensor(self.scale), requires_grad=True)
        if self.is_symmetric == False:
            self.zero_point = nn.Parameter(torch.tensor(self.zero_point+1e-9), requires_grad=True)
        else:
            self.zero_point = 0
    
    def init_scaling_factor_for_learning(self):
        self.scale = 2*np.mean(self.mean_x)/np.sqrt(2**self.bits_width-1)
   
    
    def winsorized_mean(self, x, lower=5, upper=95, sample_size=1000000):
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