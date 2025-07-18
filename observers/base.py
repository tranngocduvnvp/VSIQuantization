from abc import ABC, abstractmethod

class BaseObserver(ABC):
    @abstractmethod
    def observe(self, x):
        """Cập nhật thống kê dựa trên tensor x."""
        pass

    @abstractmethod
    def get_scale_zero_point(self):
        """Tính toán scale và zero_point dựa trên thống kê đã thu thập."""
        pass 