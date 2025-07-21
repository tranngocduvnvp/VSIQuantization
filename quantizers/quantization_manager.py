from utils.registry import CLASS_REGISTRY
from quantizers.uniform import *
from observers.minmax import *
import torch
import torch.nn as nn



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
        self.scale = 1
        self.zero_point = 0
    
    def collect_min_max(self, x):
        return torch.min(x), torch.max(x)
    
    def collect_qparameter(self, x):
        if self.is_learning_scale == False:
            scale, zero_point = self.observer(x)
            self.scale = scale
            self.zero_point = zero_point
        else:
            pass

    def quantize(self, x):
        self.collect_qparameter(x)
        return self.quantizer.quantize(x, self.scale, self.zero_point)

    def make_learn_qparameter(self, x):
        self.scale = nn.Parameter(torch.tensor(self.scale), requires_grad=True)
        if self.is_symmetric == False:
            self.zero_point = nn.Parameter(torch.tensor(self.zero_point), requires_grad=True)
        else:
            self.zero_point = 0
    
   