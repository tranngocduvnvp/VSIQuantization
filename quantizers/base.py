from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    @abstractmethod
    def quantize(self, x, scale, zero_point):
        """Lượng tử hóa tensor x với scale và zero_point."""
        pass 