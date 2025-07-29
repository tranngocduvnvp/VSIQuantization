import torch
from quantizers.base import BaseQuantizer
from utils.registry import register_class
from torch.autograd import Function


@register_class
class UniformQuantizer(BaseQuantizer):
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
        # print(self.calib_grad_scale)
        if is_learning_scale == True:    
            grad_scale = self.calculate_grad_scale(x)*self.calib_grad_scale
            # grad_scale = 1
            scale = self.scale_grad_func()(scale, grad_scale)
            if self.symmetric == False:
                zero_point = self.zero_point_rounding(zero_point) # rounding zero-point
                zero_point = self.scale_grad_func()(zero_point, grad_scale) # zero-point was graded
        x_int = self.discreate_tensor(x, scale, zero_point, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) * scale
        # x_dequant = FunLSQ.apply(x, scale, grad_scale, self.qmin, self.qmax)
        return x_dequant 
    
    def calculate_grad_scale(self, quant_tensor):
        num_pos_level = self.qmax  # Max level quantization Qp
        num_elements_feature = quant_tensor.numel() # nFeature of quant_tensor
        return ((num_pos_level * num_elements_feature) ** -0.5)  # 1 / sqrt (Qn * nfeatures)
    
    def scale_grad_func(self):
        return ScaleGradient.apply
    
    def discretizer(self):
        return RoundStraightThrough.apply

    def discreate_tensor(self, x, scale, zero_point, quant_min, quant_max):
        x_int = torch.clamp(self.discretizer()(x/scale+zero_point), quant_min, quant_max)
        return x_int
    
    def zero_point_rounding(self, zero_point):
        zero_point = self.discretizer()(zero_point)
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return zero_point


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, g, Qn, Qp):
        assert scale > 0, 'alpha = {}'.format(scale)
        ctx.save_for_backward(weight, scale)
        ctx.other = g, Qn, Qp
        q_w = (weight / scale).round().clamp(Qn, Qp)
        w_q = q_w * scale
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / scale
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big # Thanks to @haolibai 
        gradient_scale = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, gradient_scale, None, None, None
    
    

def scale_grad_(x: torch.Tensor, gamma: float) -> torch.Tensor:
      
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
    @staticmethod
    def forward(ctx, input, scale=4, floor=False):
        ctx.scale = scale
        ctx.input = input.clone().detach()
        if floor == True:
            return torch.floor(input)
        return torch.round(input)
    @staticmethod
    def backward(ctx, output_grad):
        input = ctx.input.detach()
        device = input.device
        x_range = input - torch.floor(input) - 0.5
        
        y_derivative = 4 * torch.sigmoid(ctx.scale * x_range) * (1 - torch.sigmoid(ctx.scale * x_range))


        return output_grad * y_derivative, None  # scale không có gradient


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass-through gradient
        return grad_output

def sign_ste(x):
    return SignSTE.apply(x)

class ScaleGradient(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None
    
class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        # print(output_grad)
        return output_grad
