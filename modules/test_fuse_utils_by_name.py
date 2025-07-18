import torch
import torch.nn as nn
from fuse import fuse_modules_unified
import sys
sys.path.append("../")
from nets.yolov8 import yolo_v8_n


def print_model_structure(model, prefix=""): 
    for name, module in model.named_children():
        print(f"{prefix}{name}: {type(module)}")
        print_model_structure(module, prefix + "  ")

def test_fuse_by_name():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.seq = nn.Sequential(
                nn.Linear(100, 50),
                nn.BatchNorm1d(50),
                nn.ReLU()
            )
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = x.flatten(1)
            x = self.seq(x)
            return x

    model = yolo_v8_n(1)
    print("Before fuse:")
    print(model)

    # Fuse conv1+bn1+relu1, seq.0+seq.1+seq.2
    fuse_name_lists = [
        ["conv", "bn", "relu"],
    ]
    fuse_modules_unified(model, fuse_name_lists, is_trace=True)

    print("\nAfter fuse:")
    print(model)

if __name__ == "__main__":
    test_fuse_by_name() 