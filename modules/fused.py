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
    def __init__(self, gamma, beta) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x):
        # print(x.shape, self.gamma.shape, self.beta.shape)
        return self.gamma.reshape(1, -1, 1, 1)*x+self.beta.reshape(1, -1, 1, 1)

class ConvBnReLU(FakeQuantize):
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

        self.conv_fuse = nn.Conv2d(
            in_channels=cv.in_channels,
            out_channels=cv.out_channels,
            kernel_size=cv.kernel_size,
            stride=cv.stride,
            padding=cv.padding,
            dilation=cv.dilation,
            groups=cv.groups,
            bias=True  if is_fuse_bn or cv.bias is not None else False
        )
        
        self.is_fuse_bn = is_fuse_bn
        self.is_relu = isinstance(relu, nn.ReLU)
        
        weight = cv.weight.data.clone()
        bias = 0
        if cv.bias is not None:
            bias = cv.bias.data.clone()
        self.conv_fuse.weight.data.copy_(weight)
        if self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(bias)

        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        running_mean = bn.running_mean.data.clone()
        running_var = bn.running_var.data.clone()
        eps = bn.eps
        std = torch.sqrt(running_var + eps)

        if is_fuse_bn:
            #Merge weight of convolution and batchnorm convolution
            

            W_fused = weight * (gamma / std).reshape([-1, 1, 1, 1])
            b_fused = beta + (bias - running_mean) * (gamma / std)
            self.conv_fuse.weight.data.copy_(W_fused)
            if self.conv_fuse.bias is not None:
                self.conv_fuse.bias.data.copy_(b_fused)
        else:
            self.bn = bn
            # self.bn = ScaleFact((gamma/std), (beta-running_mean/std*gamma))

    def run_forward_core(self, x, weights, bias):
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
    def __init__(self, cv, bn, observer_w_name: str, quantizer_w_name: str, observer_a_name: str, quantizer_a_name: str, w_symmetric: bool = True, a_symmetric: bool = True, is_fuse_bn=True, bits_w: int = 8, bits_a: int = 8):
        super().__init__(cv, bn, None, observer_w_name, quantizer_w_name, observer_a_name, quantizer_a_name, w_symmetric, a_symmetric, is_fuse_bn, bits_w, bits_a)
    
    def run_forward_core(self, x, weights, bias):
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
            bias=True  if cv.bias is not None else False
        )
        # Copy weights from original conv
        self.conv_fuse.weight.data.copy_(cv.weight.data)
        if cv.bias is not None and self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(cv.bias.data)
        self.is_relu = isinstance(relu, nn.ReLU)
    
    def run_forward_core(self, x, weights, bias):
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
            bias=True  if cv.bias is not None else False
        )
        # Copy weights from original conv
        self.conv_fuse.weight.data.copy_(cv.weight.data)
        if cv.bias is not None and self.conv_fuse.bias is not None:
            self.conv_fuse.bias.data.copy_(cv.bias.data)
    
    def run_forward_core(self, x, weights, bias):
        x = F.conv2d(
            x, weights, bias,
            stride=self.conv_fuse.stride,
            padding=self.conv_fuse.padding,
            dilation=self.conv_fuse.dilation,
            groups=self.conv_fuse.groups 
        )
        return x
    

class LinearBnReLU(FakeQuantize):
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
            bias=True  if is_fuse_bn or linear.bias is not None else False# b?t bias
        )
        self.is_fuse_bn = is_fuse_bn
        self.is_relu = isinstance(relu, nn.ReLU)

        weight = linear.weight.data.clone()
        bias = linear.bias.data.clone()
        self.linear_fuse.weight.data.copy_(weight)
        if self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(bias)

        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        running_mean = bn.running_mean.data.clone()
        running_var = bn.running_var.data.clone()
        eps = bn.eps
        std = torch.sqrt(running_var + eps)

        if is_fuse_bn:
            #Merge weight of convolution and batchnorm convolution
            W_fused = weight * (gamma / std).reshape([-1, 1])
            b_fused = beta + (bias - running_mean) * (gamma / std)
            self.linear_fuse.weight.data.copy_(W_fused)
            if self.linear_fuse.bias is not None:
                self.linear_fuse.bias.data.copy_(b_fused)
        else:
            self.bn = bn
            # self.bn = ScaleFact(gamma/std, beta-running_mean/std*gamma)

    def run_forward_core(self, x, weights, bias):
        # weights, bias are quantized
        x = F.linear(x, weights, bias)
        if not self.is_fuse_bn:
            x = self.bn(x)
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x

    def get_weight_bias(self):
        bias = None
        if self.linear_fuse.bias is not None:
            bias = self.linear_fuse.bias
        return self.linear_fuse.weight, bias

class LinearBn(LinearBnReLU):
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
        # weights, bias are quantized
        x = F.linear(x, weights, bias)
        if not self.is_fuse_bn:
            x = self.bn(x)
        return x



class LinearReLU(FakeQuantize):
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
            bias=True  if linear.bias else False# b?t bias
        )
        # Copy weights from original linear
        self.linear_fuse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(linear.bias.data)
        self.is_relu = isinstance(relu, nn.ReLU)
    def run_forward_core(self, x, weights, bias):
        x = F.linear(x, weights, bias)
        x = F.relu(x) if self.is_relu else F.silu(x)
        return x


class Linear(FakeQuantize):
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
            bias=True  if linear.bias else False# b?t bias
        )
        # Copy weights from original linear
        self.linear_fuse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and self.linear_fuse.bias is not None:
            self.linear_fuse.bias.data.copy_(linear.bias.data)
    
    def run_forward_core(self, x, weights, bias):
        x = F.linear(x, weights, bias)
        return x
    