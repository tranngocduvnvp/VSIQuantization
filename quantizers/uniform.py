import torch
from quantizers.base import BaseQuantizer
from utils.registry import register_class
from torch.autograd import Function


@register_class
class UniformQuantizer(BaseQuantizer):
    """
    Uniform quantizer implementation with optional symmetric quantization.
    
    This quantizer maps floating point values to a uniform grid of quantized values.
    Supports both symmetric and asymmetric quantization with learnable parameters.
    
    Args:
        num_bits (int): Number of bits for quantization
        symmetric (bool): If True, use symmetric quantization around zero
        
    Attributes:
        qmin (int): Minimum quantized value
        qmax (int): Maximum quantized value
        calib_grad_scale (float): Gradient scaling factor for calibration
    """
    def __init__(self, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.qmin = 0
        self.qmax = 2 ** self.num_bits - 1
        if self.symmetric:
            self.qmin = - (2 ** (self.num_bits - 1))
            self.qmax = 2 ** (self.num_bits - 1) - 1
        self.calib_grad_scale = 1
    
    def quantize(self, x, scale, zero_point, is_learning_scale):
        """
        Quantize input tensor using scale and zero point.
        
        Args:
            x (Tensor): Input tensor to quantize
            scale (float): Scale factor for quantization
            zero_point (int): Zero point for asymmetric quantization
            is_learning_scale (bool): Whether scale/zero_point are learnable
            
        Returns:
            Tensor: Dequantized tensor after quantization
        """
        if is_learning_scale:    
            grad_scale = self.calculate_grad_scale(x) * self.calib_grad_scale
            scale = self.scale_grad_func()(scale, grad_scale)
            if not self.symmetric:
                zero_point = self.zero_point_rounding(zero_point)  # Round zero-point
                zero_point = self.scale_grad_func()(zero_point, grad_scale)  # Apply gradient scaling
        
        x_int = self.discreate_tensor(x, scale, zero_point, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant 
    
    def calculate_grad_scale(self, quant_tensor):
        """
        Calculate gradient scaling factor based on tensor properties.
        
        Args:
            quant_tensor (Tensor): Tensor being quantized
            
        Returns:
            float: Gradient scaling factor = 1/sqrt(Qp * num_elements)
            where Qp is the number of positive quantization levels
        """
        num_pos_level = self.qmax  # Max level quantization Qp
        num_elements_feature = quant_tensor.numel()  # Number of elements
        return ((num_pos_level * num_elements_feature) ** -0.5)
    
    def scale_grad_func(self):
        """Get the gradient scaling function"""
        return ScaleGradient.apply
    
    def discretizer(self):
        """Get the rounding function with straight-through gradient"""
        return RoundStraightThrough.apply

    def discreate_tensor(self, x, scale, zero_point, quant_min, quant_max):
        """
        Convert continuous tensor to discrete values.
        
        Args:
            x (Tensor): Input tensor
            scale (float): Scale factor
            zero_point (int): Zero point offset
            quant_min (int): Minimum quantized value
            quant_max (int): Maximum quantized value
            
        Returns:
            Tensor: Discretized tensor with values in [quant_min, quant_max]
        """
        x_int = torch.clamp(self.discretizer()(x/scale+zero_point), quant_min, quant_max)
        return x_int
    
    def zero_point_rounding(self, zero_point):
        """Round zero point and clamp to valid range"""
        zero_point = self.discretizer()(zero_point)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        return zero_point


class FunLSQ(torch.autograd.Function):
    """
    Custom autograd function for Learned Step-size Quantization (LSQ).
    
    Implements forward and backward passes for LSQ quantization with
    learnable scale parameter.
    """
    @staticmethod
    def forward(ctx, weight, scale, g, Qn, Qp):
        """
        Forward pass of LSQ quantization.
        
        Args:
            weight (Tensor): Input weights to quantize
            scale (float): Scale factor (must be positive)
            g (float): Gradient scaling factor
            Qn (int): Minimum quantized value
            Qp (int): Maximum quantized value
            
        Returns:
            Tensor: Quantized-dequantized weights
        """
        assert scale > 0, 'alpha = {}'.format(scale)
        ctx.save_for_backward(weight, scale)
        ctx.other = g, Qn, Qp
        q_w = (weight / scale).round().clamp(Qn, Qp)
        w_q = q_w * scale
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        """
        Custom gradient computation for LSQ.
        
        Implements gradient computation for both weights and scale factor,
        handling gradient flow through the quantization operation.
        """
        weight, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / scale
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        gradient_scale = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, gradient_scale, None, None, None


def scale_grad_(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute gradient scaling for smooth quantization.
    
    Args:
        x (Tensor): Input tensor
        gamma (float): Smoothing parameter
        
    Returns:
        Tensor: Scaled gradients for smooth quantization
    """
    left_expr = ((0.5 - gamma) / gamma) * (x + 0.5)
    middle_expr = -x
    right_expr = ((0.5 - gamma) / gamma) * (x - 0.5)
    
    return torch.where(
        (x >= -0.5 - gamma) & (x <= -0.5 + gamma), left_expr,
        torch.where(
            (x > -0.5 + gamma) & (x <= 0.5 - gamma), middle_expr,
            torch.where(
                (x > 0.5 - gamma) & (x <= 0.5 + gamma), right_expr,
                torch.full_like(x, float('nan'))
            )
        )
    )


class Smooth_rounding(Function):
    """
    Differentiable rounding function with smooth gradients.
    
    Uses sigmoid-based smoothing for better gradient propagation
    through the rounding operation.
    """
    @staticmethod
    def forward(ctx, input, scale=4, floor=False):
        """
        Forward pass of smooth rounding.
        
        Args:
            input (Tensor): Input tensor
            scale (float): Smoothing scale factor
            floor (bool): Whether to use floor instead of round
            
        Returns:
            Tensor: Rounded tensor
        """
        ctx.scale = scale
        ctx.input = input.clone().detach()
        if floor:
            return torch.floor(input)
        return torch.round(input)
        
    @staticmethod
    def backward(ctx, output_grad):
        """
        Compute smooth gradients for rounding.
        
        Uses sigmoid-based smoothing to compute gradients that
        better approximate the rounding operation.
        """
        input = ctx.input.detach()
        x_range = input - torch.floor(input) - 0.5
        y_derivative = 4 * torch.sigmoid(ctx.scale * x_range) * (1 - torch.sigmoid(ctx.scale * x_range))
        return output_grad * y_derivative, None


class SignSTE(torch.autograd.Function):
    """
    Sign function with straight-through gradient estimator.
    
    Forward: Computes sign of input
    Backward: Passes gradient straight through
    """
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign_ste(x):
    """Helper function to apply sign with straight-through estimator"""
    return SignSTE.apply(x)


class ScaleGradient(Function):
    """
    Custom function to scale gradients during backpropagation.
    
    Useful for controlling gradient magnitude in quantization-aware training.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None


class RoundStraightThrough(Function):
    """
    Rounding operation with straight-through gradient estimator.
    
    Forward: Rounds input to nearest integer
    Backward: Passes gradient straight through
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad
