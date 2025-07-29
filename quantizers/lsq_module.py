from torch.quantization import FakeQuantize
from torch.quantization import MovingAverageMinMaxObserver
import torch
from torch.autograd import Function
import torch.nn as nn
import random
import numpy


def setup_seed():
    """
    Setup deterministic random seeds for reproducibility.
    
    Sets fixed seeds for Python's random, NumPy, and PyTorch (CPU and CUDA)
    to ensure experiments can be reproduced exactly.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_interp_nd(input, xp_template, fp_template):
    """
    N-dimensional linear interpolation similar to np.interp for PyTorch tensors.
    
    Args:
        input (Tensor): Input tensor of any shape (...) to interpolate
        xp_template (Tensor): 1D tensor (N,) of x-coordinates for interpolation points
        fp_template (Tensor): 1D tensor (N,) of y-coordinates for interpolation points
        
    Returns:
        Tensor: Interpolated values with same shape as input
        
    Note:
        This function performs linear interpolation independently on each element
        of the input tensor using the provided interpolation points.
    """
    device = input.device
    input_flat = input.reshape(-1)  # (numel,)
    numel = input_flat.shape[0]
    N = xp_template.shape[0]

    # Broadcast interpolation points for each input element
    xp = xp_template.view(1, -1).to(device)  # (1, N)
    fp = fp_template.view(1, -1).to(device)  # (1, N)

    # Floor each element
    floor_input = torch.floor(input_flat).unsqueeze(1)  # (numel, 1)
    xp_all = xp + floor_input + 0.5  # (numel, N), separate range for each input

    x_expanded = input_flat.unsqueeze(1).expand(-1, N)  # (numel, N)

    # Find indices where xp[j] <= x < xp[j+1]
    idx = torch.sum(xp_all <= x_expanded, dim=1).clamp(1, N - 1)  # (numel,)

    # Get interpolation points
    x0 = xp_all[torch.arange(numel), idx - 1]
    x1 = xp_all[torch.arange(numel), idx]
    y0 = fp.expand(numel, -1)[torch.arange(numel), idx - 1]
    y1 = fp.expand(numel, -1)[torch.arange(numel), idx]

    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (input_flat - x0)

    return y.view(input.shape)  # Reshape to original dimensions


class LSQFakeQuantize(FakeQuantize):
    """
    Learned Step Size Quantization (LSQ) with learnable scale and zero point.
    
    This quantizer learns optimal quantization parameters during training through
    backpropagation. It supports both per-tensor and per-channel quantization.
    
    Args:
        learn_scale (bool): Whether to learn quantization parameters
        config_act (bool): Whether this is for activation quantization
        observer: Observer class for collecting statistics
        quant_min (int): Minimum quantized value
        quant_max (int): Maximum quantized value
        **observer_kwargs: Additional arguments for the observer
    """
    def __init__(self, learn_scale=False, config_act=False, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        device = self.scale.device
        self.learn_scale = learn_scale
        self.flag_param_quant = False
        self.flag_adaptive = False
        self.config_act = config_act

    def forward(self, X):
        """
        Forward pass with quantization.
        
        If observer is enabled:
        1. Collects statistics about input tensor
        2. Updates scale and zero point
        3. Initializes learnable parameters if needed
        
        If fake quantization is enabled:
        1. Applies LSQ quantization with learned parameters
        2. Supports both per-tensor and per-channel quantization
        
        Args:
            X (Tensor): Input tensor to quantize
            
        Returns:
            Tensor: Quantized tensor
        """
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

            if self.learn_scale:
                # Initialize quantization parameters
                scale_init = _scale
                zero_point_init = _zero_point.float()
                
                if self.is_per_channel:
                    scale_init = scale_init.view([1] + [-1]+[1]*(len(X.shape)-2))
                    zero_point_init = zero_point_init.view([1] + [-1]+[1]*(len(X.shape)-2))
                
                if not self.flag_param_quant:
                    self.register_parameter("scale_param", torch.nn.Parameter(scale_init))
                    self.register_parameter("zero_point_param_float", torch.nn.Parameter(zero_point_init))
                    if not self.config_act:
                        self.register_parameter("theta", torch.nn.Parameter(torch.ones_like(X)))
                        self.register_parameter("gamma", torch.nn.Parameter(torch.zeros_like(X)))
                        self.theta.requires_grad = False
                        self.gamma.requires_grad = False
                    self.flag_param_quant = True
                else:
                    self.scale_param.data.copy_(scale_init)
                    self.zero_point_param_float.data.copy_(zero_point_init)

        if self.fake_quant_enabled[0] == 1:
            if self.learn_scale and self.observer_enabled[0] == 0:
                # Compute gradient scaling
                grad_scale = self.calculate_grad_scale(X)
                if self.config_act:
                    grad_scale *= 5000
                
                # Apply gradient scaling to parameters
                scale = self.scale_grad_func()(self.scale_param, grad_scale)
                quant_zero_point = self.zero_point_rounding()
                zero_point = self.scale_grad_func()(quant_zero_point, grad_scale)
            else:
                scale = self.scale
                zero_point = self.zero_point
                if self.is_per_channel:
                    scale = scale.view([1] + [-1]+[1]*(len(X.shape)-2))
                    zero_point = zero_point.view([1] + [-1]+[1]*(len(X.shape)-2))
            
            # Apply quantization
            if self.is_per_channel:
                X = self.fake_quantize_per_channel_affine(
                    X, scale, zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                X = self.fake_quantize_per_tensor_affine(
                    X, scale, zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)

        return X

    def scale_weight_grad(self, quant_tensor, alpha=1):
        """
        Compute gradient scaling factor for weights based on quantization error.
        
        Args:
            quant_tensor (Tensor): Tensor being quantized
            alpha (float): Scaling hyperparameter
            
        Returns:
            Tensor: Gradient scaling factors
        """
        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        scale_grad_weight = torch.exp(-alpha*torch.abs(torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy))
        return scale_grad_weight

    def scale_grad_scale_param(self, quant_tensor):
        """
        Compute gradient scaling for scale parameter based on quantization error.
        
        Args:
            quant_tensor (Tensor): Tensor being quantized
            
        Returns:
            Tensor: Gradient scaling factors for scale parameter
        """
        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        sub_round = torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy
        
        grad_scale_gamma = -self.scale_grad_(sub_round, gamma=0.1)/(sub_round+1e-30)
        grad_scale_gamma = torch.clamp(grad_scale_gamma, -2, 2)
        return grad_scale_gamma

    def activate_grad_theta(self):
        """Enable gradient computation for theta and gamma parameters"""
        self.theta.requires_grad = True
        self.gamma.requires_grad = True

    def fake_quantize_per_tensor_affine(self, x, scale, zero_point, quant_min, quant_max):
        """
        Apply per-tensor affine quantization.
        
        Args:
            x (Tensor): Input tensor
            scale (float): Scale factor
            zero_point (int): Zero point
            quant_min (int): Minimum quantized value
            quant_max (int): Maximum quantized value
            
        Returns:
            Tensor: Quantized tensor
        """
        if not self.flag_adaptive:
            x_int = self.discreate_tensor(x, scale, zero_point, quant_min, quant_max)
        else:
            x_int = self.discreate_adaptive_tensor(x, scale, zero_point, quant_min, quant_max)
        x_quant = scale*(x_int-zero_point)
        return x_quant

    def fake_quantize_per_tensor_power_of_two(self, x):
        """
        Apply power-of-two quantization.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Quantized tensor with power-of-two values
        """
        p = torch.log2(torch.abs(x))
        p = self.discretizer()(p)
        s = self.signSTE()(x)
        return s*torch.pow(2, p)

    def fake_quantize_per_channel_affine(self, x, scale, zero_point, ch_axis, quant_min, quant_max):
        """
        Apply per-channel affine quantization.
        
        Args:
            x (Tensor): Input tensor
            scale (Tensor): Scale factors per channel
            zero_point (Tensor): Zero points per channel
            ch_axis (int): Channel axis
            quant_min (int): Minimum quantized value
            quant_max (int): Maximum quantized value
            
        Returns:
            Tensor: Quantized tensor
        """
        if not self.flag_adaptive:
            x_int = self.discreate_tensor(x, scale, zero_point, quant_min, quant_max)
        else:
            x_int = self.discreate_adaptive_tensor(x, scale, zero_point, quant_min, quant_max)
        x_quant = scale*(x_int-zero_point)
        return x_quant

    def discreate_tensor(self, x, scale, zero_point, quant_min, quant_max):
        """
        Convert continuous tensor to discrete values.
        
        Args:
            x (Tensor): Input tensor
            scale (float/Tensor): Scale factor(s)
            zero_point (int/Tensor): Zero point(s)
            quant_min (int): Minimum quantized value
            quant_max (int): Maximum quantized value
            
        Returns:
            Tensor: Discretized tensor
        """
        x_int = torch.clamp(self.discretizer()(x/scale+zero_point), quant_min, quant_max)
        return x_int

    def discreate_adaptive_tensor(self, x, scale, zero_point, quant_min, quant_max):
        """
        Convert continuous tensor to discrete values with adaptive rounding.
        
        Uses learned parameters theta and gamma to adjust the rounding behavior
        for potentially better quantization.
        
        Args:
            x (Tensor): Input tensor
            scale (float/Tensor): Scale factor(s)
            zero_point (int/Tensor): Zero point(s)
            quant_min (int): Minimum quantized value
            quant_max (int): Maximum quantized value
            
        Returns:
            Tensor: Discretized tensor with adaptive rounding
        """
        x.requires_grad = False
        affine_ = self.theta
        h_theta = torch.clamp(torch.tanh(affine_)*1.2, -1, 1)
        self.saved_h_theta = h_theta
        x_q = torch.clamp(self.discretizer()(x/scale+zero_point) + h_theta, quant_min, quant_max)
        return x_q

    def calculate_grad_scale(self, quant_tensor):
        """
        Calculate gradient scaling factor based on tensor properties.
        
        Args:
            quant_tensor (Tensor): Tensor being quantized
            
        Returns:
            float: Gradient scaling factor = 1/sqrt(Qp * num_elements)
        """
        num_pos_level = self.quant_max  # Max quantization level
        num_elements_feature = quant_tensor.numel()  # Number of elements
        if self.is_per_channel:
            num_elements_feature /= quant_tensor.shape[1]
        
        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        sub_round = torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy
        
        grad_scale_gamma = -torch.sum(self.scale_grad_(sub_round, gamma=0.1))/(torch.sum(sub_round)+1e-10)
        grad_scale_gamma = torch.clamp(grad_scale_gamma, -2, 2)

        return ((num_pos_level * num_elements_feature) ** -0.5)

    def scale_grad_func(self):
        """Get the gradient scaling function"""
        return ScaleGradient.apply

    def discretizer(self):
        """Get the rounding function with straight-through gradient"""
        return RoundStraightThrough.apply

    def signSTE(self):
        """Get the sign function with straight-through gradient"""
        return SignSTE.apply

    def zero_point_rounding(self):
        """Round zero point and clamp to valid range"""
        zero_point = self.discretizer()(self.zero_point_param_float)
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return zero_point

    def scale_grad_(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute gradient scaling for smooth quantization.
        
        Args:
            x (Tensor): Input tensor
            gamma (float): Smoothing parameter
            
        Returns:
            Tensor: Scaled gradients
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


if __name__ == "__main__":
    """Example usage of LSQ quantization with a simple CNN model"""
    
    setup_seed()

    class CNNModel(nn.Module):
        """Simple CNN model for testing quantization"""
        def __init__(self):
            super().__init__() 
            self.conv2d = nn.Conv2d(1, 2, 3)
            self.relu = nn.ReLU()
            self.ln = nn.Linear(2, 2)
            
        def forward(self, x):
            x = self.relu(self.conv2d(x))
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.view(x.shape[0], -1)
            x = self.ln(x)
            return x

    device = "cuda"
    model = CNNModel().to(device)
    dumpy_batch = torch.randn(2, 1, 6, 6).to(device)
    dumpy_label = torch.randint(0, 2, (2,)).to(device)

    out = model(dumpy_batch)

    # Configure quantization
    my_qconfig = torch.quantization.QConfig(
        activation=LSQFakeQuantize.with_args(
            learn_scale=True,
            observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
            quant_min=0, quant_max=2**(8)-1,
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            reduce_range=False,
            averaging_constant=0.01,
            ch_axis=1
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-(2**(8-1)), quant_max=2**(8-1)-1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        )
    )

    # Apply quantization
    model.qconfig = my_qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    model.eval()

    print("++++++++++++++++++++++")
    for _ in range(14):
        dumpy_batch = torch.randn(4, 1, 6, 6).to(device)
        __ = model(dumpy_batch)
    
    model.apply(torch.quantization.disable_observer)

    print("+++++++++++++++++++++")
    
    print(model.conv2d.activation_post_process.scale_param)
    print(model.conv2d.activation_post_process.zero_point_param_float)

    print(model.conv2d.activation_post_process.scale_param.shape)
    print(model.conv2d.activation_post_process.zero_point_param_float.shape)