from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    @abstractmethod
    def quantize(self, x, scale, zero_point):
        """Quantize tensor x with scale and zero_point."""
        pass 