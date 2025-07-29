from abc import ABC, abstractmethod
import torch.nn as nn


class BaseObserver(ABC):
    """
    Abstract base class for quantization observers.
    
    Observers collect statistics about tensor values during the forward pass
    to determine appropriate quantization parameters (scale and zero point).
    
    All concrete observer implementations must implement:
    1. observe() - To collect statistics about tensor values
    2. get_scale_zero_point() - To compute quantization parameters
    """
    
    @abstractmethod
    def observe(self, x):
        """
        Update statistics based on observing tensor values.
        
        Args:
            x (Tensor): Input tensor to observe
            
        This method should be called during forward passes to collect
        information about the distribution of values in the tensor.
        """
        pass

    @abstractmethod
    def get_scale_zero_point(self):
        """
        Compute quantization scale and zero point from collected statistics.
        
        Returns:
            tuple: (scale, zero_point) parameters for quantization
            
        This method uses the statistics collected by observe() to determine
        appropriate quantization parameters that minimize information loss.
        """
        pass 