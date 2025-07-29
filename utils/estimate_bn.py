import torch
import torch.nn as nn
import copy
from quantizers.fake_quantize import FakeQuantize
from modules.fused import ConvBnReLU


class ReestimateBNStats:
    """
    Helper class to re-estimate batch normalization statistics.
    
    This class provides a callable interface to update running mean
    and variance statistics of batch normalization layers using
    a specified number of batches from a data loader.
    
    Args:
        model (nn.Module): Model containing BN layers to update
        data_loader: DataLoader providing batches for estimation
        num_batches (int): Number of batches to use for estimation
    """
    def __init__(self, model, data_loader, num_batches=50):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        """
        Update BN statistics when called.
        
        Args:
            engine: Ignored, included for compatibility with training engines
        """
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    """
    Re-estimate batch normalization statistics using data.
    
    This function:
    1. Sets BN momentum to 1.0 to use current batch statistics
    2. Runs forward passes on data to collect statistics
    3. Updates running mean and variance of BN layers
    4. Optionally stores EMA (exponential moving average) statistics
    
    Args:
        model (nn.Module): Model containing BN layers to update
        data_loader: DataLoader providing batches for estimation
        num_batches (int): Number of batches to use
        store_ema_stats (bool): Whether to store EMA statistics
    """
    # Set BN momentum to 1 and use train mode
    # -> running mean/var will have current batch statistics
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
                # Save original EMA in registered buffers
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
            # Save running mean/var to buffer
            for name, module in model.named_modules():
                if isinstance(module, FakeQuantize):
                    module.running_mean_sum += module.bn.running_mean
                    module.running_var_sum += module.bn.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
                
    # Normalize buffers and update running statistics
    for name, module in model.named_modules():
        if isinstance(module, ConvBnReLU):
            module.bn.running_mean = module.running_mean_sum / batch_count
            module.bn.running_var = module.running_var_sum / batch_count
            # Reset momentum to original value
            module.bn.momentum = org_momentum[name]

    model.eval()


def compute_scale(model, data_loader, num_batches=50, store_ema_stats=False):
    """
    Compute scaling factors for quantization calibration.
    
    This function:
    1. Computes weight statistics (mean, std) for each layer
    2. Computes activation statistics using BN parameters
    3. Uses these to calculate gradient scaling factors
    
    Args:
        model (nn.Module): Model to compute scales for
        data_loader: DataLoader (unused in current implementation)
        num_batches (int): Number of batches (unused)
        store_ema_stats (bool): Whether to store EMA stats (unused)
    """
    model.eval()
    
    for name, module in model.named_modules():
        if isinstance(module, ConvBnReLU):
            # Calculate weight statistics
            weight = module.conv_fuse.weight.data.detach().clone()
            mu_w = weight.mean()
            sigma_w = weight.std()
            
            # Calculate activation statistics from BN parameters
            mu_a = module.bn.bias.data.clone()
            sigma_a = module.bn.weight.data.clone()
            
            # Compute PDF and CDF for calibration
            pdf, cdf = pdf_cdf(mu_a/sigma_a, 0, 1)
            
            # Set calibration gradient scaling factor
            module.activation_quantizer.quantizer.calib_grad_scale = 1/((mu_w**2+sigma_w**2)/((mu_a**2+sigma_a**2)*cdf+\
                mu_a*sigma_a*pdf))

    model.eval()


import torch
from torch.distributions import Normal

def pdf_cdf(x, mu, sigma):
    """
    Compute PDF and CDF of normal distribution.
    
    Args:
        x (Tensor or float): Input values
        mu (float): Mean of distribution
        sigma (float): Standard deviation of distribution
        
    Returns:
        tuple: (pdf, cdf) with same dtype/device as input
            - pdf: Probability density function values
            - cdf: Cumulative distribution function values
    """
    dist = Normal(loc=mu, scale=sigma)
    pdf = torch.exp(dist.log_prob(x))  # or dist.log_prob for log-pdf
    cdf = dist.cdf(x)
    return pdf, cdf