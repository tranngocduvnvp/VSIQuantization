from abc import ABC, abstractmethod
import torch.nn as nn

class BaseObserver(ABC):
    @abstractmethod
    def observe(self, x):
        """Update statics base on observation of x."""
        pass

    @abstractmethod
    def get_scale_zero_point(self):
        """compute scale and zero_point base on statics was collected."""
        pass 