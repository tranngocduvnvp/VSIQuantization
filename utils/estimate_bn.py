import torch
import torch.nn as nn
import copy
from quantizers.fake_quantize import FakeQuantize
from modules.fused import ConvBnReLU

class ReestimateBNStats:
    def __init__(self, model, data_loader, num_batches=50):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    # We set BN momentum to 1 an use train mode
    # -> the running mean/var have the current batch statistics
    model.eval()
    org_momentum = {}
    for name, module in model.named_modules():
        if isinstance(module, ConvBnReLU):
            org_momentum[name] = module.bn.momentum
            module.bn.momentum = 1.0
            module.running_mean_sum = torch.zeros_like(module.bn.running_mean)
            module.running_var_sum = torch.zeros_like(module.bn.running_var)
            
            module.bn.training = True

            if store_ema_stats:
                # Save the original EMA, make sure they are in buffers so they end in the state dict
                if not hasattr(module, "running_mean_ema"):
                    module.register_buffer("running_mean_ema", copy.deepcopy(module.bn.running_mean))
                    module.register_buffer("running_var_ema", copy.deepcopy(module.bn.running_var))
                else:
                    module.running_mean_ema = copy.deepcopy(module.bn.running_mean)
                    module.running_var_ema = copy.deepcopy(module.bn.running_var)

    # Run data for estimation
    device = next(model.parameters()).device
    batch_count = 0
    with torch.no_grad():
        for batch_i, (imgs, targets) in (enumerate(data_loader)):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            model(imgs)
            # We save the running mean/var to a buffer
            for name, module in model.named_modules():
                if isinstance(module, FakeQuantize):
                    module.running_mean_sum += module.bn.running_mean
                    module.running_var_sum += module.bn.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
    # At the end we normalize the buffer and write it into the running mean/var
    for name, module in model.named_modules():
        if isinstance(module, ConvBnReLU):
            module.bn.running_mean = module.running_mean_sum / batch_count
            module.bn.running_var = module.running_var_sum / batch_count
            # We reset the momentum in case it would be used anywhere else
            module.bn.momentum = org_momentum[name]
            


    model.eval()

def compute_scale(model, data_loader, num_batches=50, store_ema_stats=False):
    # We set BN momentum to 1 an use train mode
    # -> the running mean/var have the current batch statistics
    model.eval()
    # org_momentum = {}
    # for name, module in model.named_modules():
    #     if isinstance(module, ConvBnReLU):
    #         org_momentum[name] = module.bn.momentum
    #         module.bn.momentum = 1.0
    #         module.running_mean_sum = torch.zeros_like(module.bn.running_mean)
    #         module.running_var_sum = torch.zeros_like(module.bn.running_var)
            
    #         module.bn.training = True

    #         if store_ema_stats:
    #             # Save the original EMA, make sure they are in buffers so they end in the state dict
    #             if not hasattr(module, "running_mean_ema"):
    #                 module.register_buffer("running_mean_ema", copy.deepcopy(module.bn.running_mean))
    #                 module.register_buffer("running_var_ema", copy.deepcopy(module.bn.running_var))
    #             else:
    #                 module.running_mean_ema = copy.deepcopy(module.bn.running_mean)
    #                 module.running_var_ema = copy.deepcopy(module.bn.running_var)

    # # Run data for estimation
    # device = next(model.parameters()).device
    # batch_count = 0
    # with torch.no_grad():
    #     for batch_i, (imgs, targets) in (enumerate(data_loader)):
    #         imgs = imgs.to(device, non_blocking=True).float() / 255.0
    #         model(imgs)
    #         # We save the running mean/var to a buffer
    #         for name, module in model.named_modules():
    #             if isinstance(module, FakeQuantize):
    #                 module.running_mean_sum += module.bn.running_mean
    #                 module.running_var_sum += module.bn.running_var

    #         batch_count += 1
    #         if batch_count == num_batches:
    #             break
    # At the end we normalize the buffer and write it into the running mean/var
    for name, module in model.named_modules():
        if isinstance(module, ConvBnReLU):
            # module.bn.running_mean = module.running_mean_sum / batch_count
            # module.bn.running_var = module.running_var_sum / batch_count
            # # We reset the momentum in case it would be used anywhere else
            # module.bn.momentum = org_momentum[name]
            
            # We calculate mean and std of weight and activation after relu
            #========== weight ========
            weight = module.conv_fuse.weight.data.detach().clone()
            mu_w = weight.mean()
            sigma_w = weight.std()
            #======== activation ======
            mu_a = module.bn.bias.data.clone()
            sigma_a = module.bn.weight.data.clone()
            pdf, cdf = pdf_cdf(mu_a/sigma_a, 0, 1)
            module.activation_quantizer.quantizer.calib_grad_scale = 1/((mu_w**2+sigma_w**2)/((mu_a**2+sigma_a**2)*cdf+\
                mu_a*sigma_a*pdf))


    model.eval()


import torch
from torch.distributions import Normal

def pdf_cdf(x, mu, sigma):
    """
    x, mu, sigma có thể là số (float) hoặc tensor PyTorch.
    Trả về (pdf, cdf) cùng kiểu dtype/device với đầu vào.
    """
    dist = Normal(loc=mu, scale=sigma)
    pdf = torch.exp(dist.log_prob(x))  # hoặc dist.log_prob → log-pdf
    cdf = dist.cdf(x)
    return pdf, cdf