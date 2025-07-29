from torch.quantization import FakeQuantize
from torch.quantization import MovingAverageMinMaxObserver
import torch
from torch.autograd import Function
import torch.nn as nn
import random
import numpy


def setup_seed():
    """
    Setup random seed.
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
    Áp d?ng n?i suy tuy?n tính gi?ng np.interp cho tensor ND (nhi?u chi?u).
    
    Args:
        input: Tensor có shape tùy ý (...), là các giá tr? c?n n?i suy.
        xp_template: Tensor 1D (N,) là các di?m m?c (ví d?: torch.linspace(-0.5, 0.5, N))
        fp_template: Tensor 1D (N,) là giá tr? tuong ?ng t?i xp

    Returns:
        Tensor có cùng shape v?i input, ch?a các giá tr? n?i suy
    """
    device = input.device
    input_flat = input.reshape(-1)  # (numel,)
    numel = input_flat.shape[0]
    N = xp_template.shape[0]

    # Broadcast xp và fp cho t?ng ph?n t? input
    xp = xp_template.view(1, -1).to(device)  # (1, N)
    fp = fp_template.view(1, -1).to(device)  # (1, N)

    # Floor theo t?ng ph?n t?
    floor_input = torch.floor(input_flat).unsqueeze(1)  # (numel, 1)
    xp_all = xp + floor_input + 0.5  # (numel, N), m?i hàng là 1 do?n m?c riêng cho t?ng input

    x_expanded = input_flat.unsqueeze(1).expand(-1, N)  # (numel, N)

    # Tìm ch? s?: xp[j] <= x < xp[j+1]
    idx = torch.sum(xp_all <= x_expanded, dim=1).clamp(1, N - 1)  # (numel,)

    x0 = xp_all[torch.arange(numel), idx - 1]
    x1 = xp_all[torch.arange(numel), idx]
    y0 = fp.expand(numel, -1)[torch.arange(numel), idx - 1]
    y1 = fp.expand(numel, -1)[torch.arange(numel), idx]

    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (input_flat - x0)

    return y.view(input.shape)  # Reshape v? shape g?c


class LSQFakeQuantize(FakeQuantize):
    def __init__(self, learn_scale=False, config_act = False, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        device = self.scale.device
        self.learn_scale = learn_scale
        self.flag_param_quant = False
        self.flag_adaptive = False
        self.config_act = config_act
        

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

            if self.learn_scale == True:
                # init params quant for LSQ
                scale_init = _scale
                zero_point_init = _zero_point.float()
                
                if self.is_per_channel:
                    scale_init = scale_init.view([1] + [-1]+[1]*(len(X.shape)-2))
                    zero_point_init = zero_point_init.view([1] + [-1]+[1]*(len(X.shape)-2))
                
                if self.flag_param_quant == False:
                    self.register_parameter("scale_param", torch.nn.Parameter(scale_init))
                    self.register_parameter("zero_point_param_float", torch.nn.Parameter(zero_point_init))
                    if self.config_act == False:
                        # print(X.shape)
                        self.register_parameter("theta", torch.nn.Parameter(torch.ones_like(X)))
                        self.register_parameter("gamma", torch.nn.Parameter(torch.zeros_like(X)))
                        self.theta.requires_grad = False
                        self.gamma.requires_grad = False
                    self.flag_param_quant = True
                
                else:
                    self.scale_param.data.copy_(scale_init)
                    self.zero_point_param_float.data.copy_(zero_point_init)



        if self.fake_quant_enabled[0] == 1:
            if self.learn_scale == True and self.observer_enabled[0] == 0:

                grad_scale = self.calculate_grad_scale(X) # compute scale for gradient
                if self.config_act == True:
                    grad_scale *=5000
             
                scale = self.scale_grad_func()(self.scale_param, grad_scale) # scale was graded
                
                # print(grad_scale.shape)

                quant_zero_point = self.zero_point_rounding() # rounding zero-point
                zero_point = self.scale_grad_func()(quant_zero_point, grad_scale) # zero-point was graded
            
            else:
                scale = self.scale
                zero_point = self.zero_point
                if self.is_per_channel:
                    scale = scale.view([1] + [-1]+[1]*(len(X.shape)-2))
                    zero_point = zero_point.view([1] + [-1]+[1]*(len(X.shape)-2))

            # if self.config_act == False:
            #     scale_grad_weight = self.scale_weight_grad(X)
            #     X = self.scale_grad_func()(X, scale_grad_weight)
            
            if self.is_per_channel:
                X = self.fake_quantize_per_channel_affine(
                    X, scale, zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                # if self.config_act == True:
                X = self.fake_quantize_per_tensor_affine(
                    X, scale, zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
                # else: 
                #     X = self.fake_quantize_per_tensor_power_of_two(X)
                
            

        return X

    def scale_weight_grad(self, quant_tensor, alpha=1):

        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        scale_grad_weight = torch.exp(-alpha*torch.abs(torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy))
        return scale_grad_weight
    

    def scale_grad_scale_param(self, quant_tensor):

        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        sub_round = torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy
        
        grad_scale_gamma = -self.scale_grad_(sub_round, gamma=0.1)/(sub_round+1e-30)
        grad_scale_gamma = torch.clamp(grad_scale_gamma, -2,2)
        return grad_scale_gamma


    def activate_grad_theta(self):
        self.theta.requires_grad = True
        self.gamma.requires_grad = True
    
    
    def fake_quantize_per_tensor_affine(self, x, scale, zero_point, quant_min, quant_max):
        if self.flag_adaptive == False:
            x_int = self.discreate_tensor(x, scale, zero_point, quant_min, quant_max)
        else:
            x_int = self.discreate_adaptive_tensor(x, scale, zero_point, quant_min, quant_max)
        x_quant = scale*(x_int-zero_point)
        return x_quant
    
    def fake_quantize_per_tensor_power_of_two(self, x):
        p = torch.log2(torch.abs(x))
        p = self.discretizer()(p)
        s = self.signSTE()(x)
        return s*torch.pow(2, p)

    def fake_quantize_per_channel_affine(self, x, scale, zero_point, ch_axis, quant_min, quant_max):
        # print(f"x.shape: {x.shape} | scale.shape: {scale.shape} | zero_point.shape: {zero_point.shape}")
        if self.flag_adaptive == False:
            x_int = self.discreate_tensor(x, scale, zero_point, quant_min, quant_max)
        else:
            x_int = self.discreate_adaptive_tensor(x, scale, zero_point, quant_min, quant_max)
        x_quant = scale*(x_int-zero_point)
        return x_quant
    
    def discreate_tensor(self, x, scale, zero_point, quant_min, quant_max):
        x_int = torch.clamp(self.discretizer()(x/scale+zero_point), quant_min, quant_max)
        return x_int

    def discreate_adaptive_tensor(self, x, scale, zero_point, quant_min, quant_max):
        #turn-off grad of tensor weight x
        x.requires_grad = False
        # Compute [x/s] 
        # affine_ = self.theta*x+self.gamma
        affine_ = self.theta
        # print(affine_.requires_grad)
        # h_theta = torch.clamp(torch.sigmoid(affine_)*1.2-0.2, 0, 1)
        h_theta = torch.clamp(torch.tanh(affine_)*1.2,-1, 1)
        # print(self.theta.grad)
        self.saved_h_theta = h_theta
        x_q = torch.clamp(self.discretizer()(x/scale+zero_point) + h_theta, quant_min, quant_max)
        return x_q
    

    def calculate_grad_scale(self, quant_tensor):
        num_pos_level = self.quant_max  # Max level quantization Qp
        num_elements_feature = quant_tensor.numel() # nFeature of quant_tensor
        if self.is_per_channel:
            # In the per tensor case we do not sum the gradients over the output channel dimension
            num_elements_feature /= quant_tensor.shape[1]
        
        quant_tensor_copy = quant_tensor.clone().detach()
        scale_param_copy = self.scale_param.clone().detach()
        
        sub_round = torch.round(quant_tensor_copy/scale_param_copy) - quant_tensor_copy/scale_param_copy
        
        grad_scale_gamma = -torch.sum(self.scale_grad_(sub_round, gamma=0.1))/(torch.sum(sub_round)+1e-10)
        grad_scale_gamma = torch.clamp(grad_scale_gamma, -2,2)
        # print(grad_scale_gamma)

        return ((num_pos_level * num_elements_feature) ** -0.5)  # 1 / sqrt (Qn * nfeatures)
    
    def scale_grad_func(self):
        return ScaleGradient.apply
    
    def discretizer(self):
        return RoundStraightThrough.apply
        # return Smooth_rounding.apply
    
    def signSTE(self):
        return SignSTE.apply
    
    def zero_point_rounding(self):
        zero_point = self.discretizer()(self.zero_point_param_float)
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return zero_point

    def scale_grad_(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
      
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


if __name__ =="__main__":
    
    setup_seed()

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__() 
            self.conv2d = nn.Conv2d(1,2,3)
            self.relu = nn.ReLU()
            self.ln = nn.Linear(2,2)
            
        def forward(self, x):
            x = self.relu(self.conv2d(x))
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.view(x.shape[0],-1)
            x = self.ln(x)
            return x

    device = "cuda"
    model = CNNModel().to(device)
    dumpy_batch = torch.randn(2,1,6,6).to(device)
    dumpy_label = torch.randint(0,2, (2, )).to(device)

    out = model(dumpy_batch)

    my_qconfig = torch.quantization.QConfig(
        activation=LSQFakeQuantize.with_args(
            learn_scale=True,
            observer=torch.quantization.MovingAveragePerChannelMinMaxObserver ,
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
            reduce_range=False,
            # averaging_constant=0.01
            )
    )

    model.qconfig = my_qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    model.eval()

    print("++++++++++++++++++++++")
    for _ in range(14):
        dumpy_batch = torch.randn(4,1,6,6).to(device)
        __ = model(dumpy_batch)
    
    model.apply(torch.quantization.disable_observer)

    print("+++++++++++++++++++++")
    
    print(model.conv2d.activation_post_process.scale_param)
    print(model.conv2d.activation_post_process.zero_point_param_float)

    print(model.conv2d.activation_post_process.scale_param.shape)
    print(model.conv2d.activation_post_process.zero_point_param_float.shape)

    # for name, value in model.named_parameters():
    #     print(name)
    
    # print("-"*30)
    # param_model= model.parameters()
    # optim = torch.optim.SGD(param_model, lr=0.01)
    # criterion = torch.nn.CrossEntropyLoss()

    # for epoch in range(3):
    #     optim.zero_grad()
    #     out = model(dumpy_batch)
    #     loss = criterion(out, dumpy_label)
    #     loss.backward() 
    #     optim.step()
    #     # print(model.conv2d.weight.grad)
    #     print(model.conv2d.activation_post_process.scale_param.grad)
    #     print(model.conv2d.activation_post_process.zero_point_param_float.grad)

    # print("-"*30)
    # print("scale param:", model.conv2d.activation_post_process.scale_param)
    # print("zero point param:", model.conv2d.activation_post_process.zero_point_param_float)