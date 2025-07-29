import imp
from tkinter import W
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from quantizers.fake_quantize import FakeQuantize


class ScaleFact(nn.Module):
    """
    Scale and shift transformation layer.
    
    This layer applies an affine transformation to the input using learnable
    parameters gamma (scale) and beta (shift).
    
    Args:
        gamma (float): Initial scale factor
        beta (float): Initial shift factor
    """
    def __init__(self, gamma, beta) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x):
        return self.gamma.reshape(1, -1, 1, 1)*x + self.beta.reshape(1, -1, 1, 1)


class ConvBnReLU(FakeQuantize):
    """
    Fused Convolution + BatchNorm + ReLU/SiLU layer with quantization support.
    
    This layer combines Conv2d, BatchNorm2d, and ReLU/SiLU into a single module,
    optionally fusing the BatchNorm parameters into the convolution weights and bias.
    
    Args:
        cv (nn.Conv2d): Original convolution layer
        bn (nn.BatchNorm2d): Original batch normalization layer
        relu (nn.ReLU or nn.SiLU): Original activation layer
        observer_w_name (str): Name of weight observer
        quantizer_w_name (str): Name of weight quantizer
        observer_a_name (str): Name of activation observer
        quantizer_a_name (str): Name of activation quantizer
        w_symmetric (bool): Whether to use symmetric weight quantization
        a_symmetric (bool): Whether to use symmetric activation quantization
        is_fuse_bn (bool): Whether to fuse batch norm parameters into conv
        bits_w (int): Number of bits for weight quantization
        bits_a (int): Number of bits for activation quantization
    """
    def __init__(self,
        cv, bn, relu,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        is_fuse_bn = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        super().__init__(observer_w_name, quantizer_w_name,\
             observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)

        # Create new conv layer with same params as original
        self.conv_fuse = nn.Conv2d(
            in_channels=cv.in_channels,
            out_channels=cv.out_channels,
            kernel_size=cv.kernel_size,
            stride=cv.stride,
            padding=cv.padding,
            dilation=cv.dilation,
            groups=cv.groups,
            bias=True if is_fuse_bn or cv.bias is not None else False
        )
        
        self.is_fuse_bn = is_fuse_bn
        self.is_relu = isinstance(relu, nn.ReLU)  # False means SiLU
        
        # Copy original weights and bias
        weight = cv.weight.data.clone()
        bias = 0
        if cv.bias is not None:
            bias = cv.bias.data.clone()
        self.conv_fuse.weight.data.copy_(weight)
        if self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(bias)

        # Get batch norm parameters
        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        running_mean = bn.running_mean.data.clone()
        running_var = bn.running_var.data.clone()
        eps = bn.eps
        std = torch.sqrt(running_var + eps)

        if is_fuse_bn:
            # Fuse batch norm parameters into conv weights and bias
            # W_fused = W * (gamma / sqrt(var + eps))
            # b_fused = beta + (b - mean) * (gamma / sqrt(var + eps))
            W_fused = weight * (gamma / std).reshape([-1, 1, 1, 1])
            b_fused = beta + (bias - running_mean) * (gamma / std)
            self.conv_fuse.weight.data.copy_(W_fused)
            if self.conv_fuse.bias is not None:
                self.conv_fuse.bias.data.copy_(b_fused)
        else:
            self.bn = bn

    def run_forward_core(self, x, weights, bias):
        """
        Core forward computation with quantized weights and bias.
        
        Args:
            x (Tensor): Input tensor
            weights (Tensor): Quantized weights
            bias (Tensor or None): Quantized bias if exists
            
        Returns:
            Tensor: Output after conv, (optional) bn, and activation
        """
        x = F.conv2d(
            x, weights, bias,
            stride=self.conv_fuse.stride,
            padding=self.conv_fuse.padding,
            dilation=self.conv_fuse.dilation,
            groups=self.conv_fuse.groups 
        )
        if not self.is_fuse_bn:
            x = self.bn(x)
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x


class ConvBn(ConvBnReLU):
    """
    Fused Convolution + BatchNorm layer with quantization support.
    Inherits from ConvBnReLU but without activation function.
    """
    def __init__(self, cv, bn, observer_w_name: str, quantizer_w_name: str, observer_a_name: str, quantizer_a_name: str, w_symmetric: bool = True, a_symmetric: bool = True, is_fuse_bn=True, bits_w: int = 8, bits_a: int = 8):
        super().__init__(cv, bn, None, observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, is_fuse_bn, bits_w, bits_a)
    
    def run_forward_core(self, x, weights, bias):
        """Forward pass without activation function"""
        x = F.conv2d(
            x, weights, bias,
            stride=self.conv_fuse.stride,
            padding=self.conv_fuse.padding,
            dilation=self.conv_fuse.dilation,
            groups=self.conv_fuse.groups 
        )
        if not self.is_fuse_bn:
            x = self.bn(x)
        return x


class ConvReLU(FakeQuantize):
    """
    Fused Convolution + ReLU/SiLU layer with quantization support.
    Similar to ConvBnReLU but without batch normalization.
    """
    def __init__(self,
        cv, relu,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        super().__init__(observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)
        
        self.conv_fuse = nn.Conv2d(
            in_channels=cv.in_channels,
            out_channels=cv.out_channels,
            kernel_size=cv.kernel_size,
            stride=cv.stride,
            padding=cv.padding,
            dilation=cv.dilation,
            groups=cv.groups,
            bias=True if cv.bias is not None else False
        )
        # Copy weights from original conv
        self.conv_fuse.weight.data.copy_(cv.weight.data)
        if cv.bias is not None and self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(cv.bias.data)
        self.is_relu = isinstance(relu, nn.ReLU)  # False means SiLU
    
    def run_forward_core(self, x, weights, bias):
        """Forward pass with conv and activation"""
        x = F.conv2d(
            x, weights, bias,
            stride=self.conv_fuse.stride,
            padding=self.conv_fuse.padding,
            dilation=self.conv_fuse.dilation,
            groups=self.conv_fuse.groups 
        )
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x


class Conv(FakeQuantize):
    """
    Quantization-aware Convolution layer without fusion.
    Basic conv layer with quantization support.
    """
    def __init__(self,
        cv,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        super().__init__(observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)
        
        self.conv_fuse = nn.Conv2d(
            in_channels=cv.in_channels,
            out_channels=cv.out_channels,
            kernel_size=cv.kernel_size,
            stride=cv.stride,
            padding=cv.padding,
            dilation=cv.dilation,
            groups=cv.groups,
            bias=True if cv.bias is not None else False
        )
        # Copy weights from original conv
        self.conv_fuse.weight.data.copy_(cv.weight.data)
        if cv.bias is not None and self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(cv.bias.data)
    
    def run_forward_core(self, x, weights, bias):
        """Simple convolution forward pass"""
        x = F.conv2d(
            x, weights, bias,
            stride=self.conv_fuse.stride,
            padding=self.conv_fuse.padding,
            dilation=self.conv_fuse.dilation,
            groups=self.conv_fuse.groups 
        )
        return x


class LinearBnReLU(FakeQuantize):
    """
    Fused Linear + BatchNorm + ReLU/SiLU layer with quantization support.
    
    Similar to ConvBnReLU but for linear (fully connected) layers.
    Supports optional batch norm fusion and choice of ReLU/SiLU activation.
    """
    def __init__(self, 
                 linear, 
                 bn, relu,
                 observer_w_name: str, 
                 quantizer_w_name: str, 
                 observer_a_name: str, 
                 quantizer_a_name: str, 
                 w_symmetric: bool = True, 
                 a_symmetric: bool = True, 
                 is_fuse_bn: bool = True,
                 bits_w: int = 8,
                 bits_a: int = 8):
        super().__init__(observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)
        self.linear_fuse = nn.Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=True if is_fuse_bn or linear.bias is not None else False
        )
        self.is_fuse_bn = is_fuse_bn
        self.is_relu = isinstance(relu, nn.ReLU)  # False means SiLU

        # Copy original weights and bias
        weight = linear.weight.data.clone()
        bias = linear.bias.data.clone()
        self.linear_fuse.weight.data.copy_(weight)
        if self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(bias)

        # Get batch norm parameters
        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        running_mean = bn.running_mean.data.clone()
        running_var = bn.running_var.data.clone()
        eps = bn.eps
        std = torch.sqrt(running_var + eps)

        if is_fuse_bn:
            # Fuse batch norm parameters into linear weights and bias
            W_fused = weight * (gamma / std).reshape([-1, 1])
            b_fused = beta + (bias - running_mean) * (gamma / std)
            self.linear_fuse.weight.data.copy_(W_fused)
            if self.linear_fuse.bias is not None:
                self.linear_fuse.bias.data.copy_(b_fused)
        else:
            self.bn = bn

    def run_forward_core(self, x, weights, bias):
        """Forward pass with quantized weights"""
        x = F.linear(x, weights, bias)
        if not self.is_fuse_bn:
            x = self.bn(x)
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x

    def get_weight_bias(self):
        """Get the layer's weight and bias parameters"""
        bias = None
        if self.linear_fuse.bias is not None:
            bias = self.linear_fuse.bias
        return self.linear_fuse.weight, bias


class LinearBn(LinearBnReLU):
    """
    Fused Linear + BatchNorm layer with quantization support.
    Inherits from LinearBnReLU but without activation function.
    """
    def __init__(self, 
                linear, 
                bn, 
                observer_w_name: str, 
                quantizer_w_name: str, 
                observer_a_name: str, 
                quantizer_a_name: str, 
                w_symmetric: bool = True, 
                a_symmetric: bool = True, 
                is_fuse_bn: bool = True,
                bits_w: int = 8,
                bits_a: int = 8):
        super().__init__(linear, bn, None, observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, is_fuse_bn, bits_w, bits_a)

    def run_forward_core(self, x, weights, bias):
        """Forward pass without activation"""
        x = F.linear(x, weights, bias)
        if not self.is_fuse_bn:
            x = self.bn(x)
        return x


class LinearReLU(FakeQuantize):
    """
    Fused Linear + ReLU/SiLU layer with quantization support.
    Similar to LinearBnReLU but without batch normalization.
    """
    def __init__(self,
        linear, relu,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        super().__init__(observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)
        self.linear_fuse = nn.Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=True if linear.bias else False
        )
        # Copy weights from original linear
        self.linear_fuse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(linear.bias.data)
        self.is_relu = isinstance(relu, nn.ReLU)  # False means SiLU
        
    def run_forward_core(self, x, weights, bias):
        """Forward pass with activation"""
        x = F.linear(x, weights, bias)
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x


class Linear(FakeQuantize):
    """
    Quantization-aware Linear layer without fusion.
    Basic linear layer with quantization support.
    """
    def __init__(self,
        linear,
        observer_w_name: str,
        quantizer_w_name: str,
        observer_a_name: str,
        quantizer_a_name: str,
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        super().__init__(observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, bits_w, bits_a)
        self.linear_fuse = nn.Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=True if linear.bias else False
        )
        # Copy weights from original linear
        self.linear_fuse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(linear.bias.data)
    
    def run_forward_core(self, x, weights, bias):
        """Simple linear forward pass"""
        x = F.linear(x, weights, bias)
        return x
    