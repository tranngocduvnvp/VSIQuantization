from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """
    Abstract base class for tensor quantization implementations.
    
    Quantizers convert floating point tensors to quantized values using
    a scale factor and zero point. Different quantization schemes can be
    implemented by subclassing this base class.
    """
    
    @abstractmethod
    def quantize(self, x, scale, zero_point):
        """
        Quantize a tensor to reduced precision.
        
        Args:
            x (Tensor): Input tensor to quantize
            scale (float): Scale factor to multiply quantized values by
                         to convert back to floating point
            zero_point (int): Value that represents 0 in the quantized space
            
        Returns:
            Tensor: The quantized tensor
            
        The quantization formula is typically:
            quantized = round(x / scale) + zero_point
            
        And the dequantization formula is:
            dequantized = (quantized - zero_point) * scale
        """
        pass 