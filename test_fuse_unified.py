
import torch
import torch.nn as nn
from modules.fuse import fuse_modules_unified
import sys
from nets.yolov8 import yolo_v8_n

def print_model_structure(model, prefix=""): 
    for name, module in model.named_children():
        print(f"{prefix}{name}: {type(module)}")
        print_model_structure(module, prefix + "  ")

def test_fuse_trace():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(3,8,3)
            self.linear1 = nn.Linear(100, 50)
            self.bn2 = nn.BatchNorm1d(50)
            self.relu2 = nn.ReLU()
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = x.flatten(1)
            x = self.linear1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            return x

    # model = yolo_v8_n(1)
    model = MyModel()
    print("Before fuse:")
    # print(model)

    fuse_patterns = [
        ["conv", "bn", "relu"],
        ["linear", "bn", "relu"],
        ["conv"],
    ]
   
    fuse_modules_unified(model, fuse_patterns, is_trace=False)

    print("\nAfter fuse:")
    print(model)

if __name__ == "__main__":
    test_fuse_trace()
