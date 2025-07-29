import imp
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from quantizers.quantization_manager import QuantizationManager

class FakeQuantize(nn.Module):
    def __init__(self,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        bits_w: int=4,
        bits_a: int=8,
        quantize_out:bool=True,
        quantize_inp:bool=False,
    ):
        super().__init__()
        self.weight_quantizer = QuantizationManager(
            quantizer_w_name,
            observer_w_name,
            bits_w,
            w_symmetric,
            is_learning_scale=True
        )

        self.activation_quantizer = QuantizationManager(
            quantizer_a_name,
            observer_a_name,
            bits_a,
            a_symmetric,
            is_learning_scale=True
        )

        self.bits_w = bits_w
        self.bits_a = bits_a
        self.quantize_out = quantize_out
        self.quantize_inp = quantize_inp
    
    def forward(self, x):
        if self.quantize_inp:
            x = self.quantize_activation(x)
        weights, bias = self.get_weight_bias()
        weights = self.quantize_weights(weights)
        out = self.run_forward_core(x, weights, bias)
        if self.quantize_out:
            out = self.quantize_activation(out)
        return out

    def run_forward_core(self, x, weights, bias):
        pass

    def get_weight_bias(self):
        bias = None
        if self.conv_fuse.bias is not None:
                bias = self.conv_fuse.bias
        return self.conv_fuse.weight, bias
    
    def quantize_weights(self, weights):
        return self.weight_quantizer.quantize(weights)
    
    def quantize_activation(self, out):
        return self.activation_quantizer.quantize(out)

     

    