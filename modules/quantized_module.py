import torch
import torch.nn as nn

class QuantizedModule(nn.Module):
    def __init__(self, module, quantizer, observer, quantize_weight=True, quantize_activation=True):
        super().__init__()
        self.module = module
        self.quantizer = quantizer
        self.observer = observer
        self.quantize_weight = quantize_weight
        self.quantize_activation = quantize_activation
        self.weight_scale = None
        self.weight_zero_point = None

    def forward(self, x):
        # Lượng tử hóa trọng số nếu cần
        if self.quantize_weight and self.weight_scale is not None and self.weight_zero_point is not None:
            quantized_weight = self.quantizer.quantize(self.module.weight, self.weight_scale, self.weight_zero_point)
            self.module.weight.data = quantized_weight

        # Quan sát activation
        if self.quantize_activation:
            self.observer.observe(x)
            scale, zero_point = self.observer.get_scale_zero_point()
            x = self.quantizer.quantize(x, scale, zero_point)
        return self.module(x)

    def freeze_weight_observer(self, num_bits=8):
        # Dùng observer để tính scale/zero_point cho weight, sau đó không cập nhật nữa
        self.observer.observe(self.module.weight.data)
        self.weight_scale, self.weight_zero_point = self.observer.get_scale_zero_point(num_bits=num_bits) 