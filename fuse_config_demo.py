#!/usr/bin/env python3
"""
Demo script showing how to use the flexible fuse configuration system.
"""

import torch
import torch.nn as nn
from modules.fuse import fuse_modules_unified
from modules.fuse_config import (
    FuseConfig, 
    FuseConfigManager, 
    load_fuse_config_from_yaml,
    create_fuse_config_manager
)

from utils.quantize_manager import (
    calibrate_qat_model,
    activate_learning_qparam,
    deactivate_learning_qparam,
    activate_quantizer,
    deactivate_quantizer
)
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ð?m b?o k?t qu? reproducible (dù có th? hoi ch?m)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleModel(nn.Module):
    """Simple model for demonstration purposes."""
    def __init__(self):
        super().__init__()
        self.backbone_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.backbone_bn1 = nn.BatchNorm2d(64)
        self.backbone_relu1 = nn.ReLU()
        
        self.backbone_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.backbone_bn2 = nn.BatchNorm2d(128)
        self.backbone_relu2 = nn.ReLU()
        
        self.pl = nn.AdaptiveAvgPool2d(1)
        self.head_linear1 = nn.Linear(128, 512)
        self.head_bn1 = nn.BatchNorm1d(512)
        self.head_relu1 = nn.ReLU()
        
        self.head_linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Backbone
        y = self.backbone_relu1(self.backbone_bn1(self.backbone_conv1(x)))
        x = self.backbone_relu2(self.backbone_bn2(self.backbone_conv2(y)))
        x = self.pl(x)
        # Head
        x = x.view(x.size(0), -1)
        x = self.head_relu1(self.head_bn1(self.head_linear1(x)))
        x = self.head_linear2(x)
        return x


def demo_programmatic_config():
    """Demo using programmatic configuration."""
    print("=== Demo 1: Programmatic Configuration ===")
    
    # Create model
    model = SimpleModel()
    print(f"Original model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create config manager with custom configurations
    config_manager = create_fuse_config_manager(
        default_config=FuseConfig(
            observer_w_name="MinMaxObserver",
            quantizer_w_name="UniformQuantizer",
            observer_a_name="MinMaxObserver",
            quantizer_a_name="UniformQuantizer",
            w_symmetric=True,
            a_symmetric=True,
            is_fuse_bn=True
        ),
        layer_configs={
            "backbone.*conv": FuseConfig(
                observer_w_name="MinMaxObserver",
                quantizer_w_name="UniformQuantizer",
                observer_a_name="MinMaxObserver",
                quantizer_a_name="UniformQuantizer",
                w_symmetric=False,
                a_symmetric=True,
                is_fuse_bn=True
            ),
            "head.*linear": FuseConfig(
                observer_w_name="MinMaxObserver",
                quantizer_w_name="UniformQuantizer",
                observer_a_name="MinMaxObserver",
                quantizer_a_name="UniformQuantizer",
                w_symmetric=True,
                a_symmetric=True,
                is_fuse_bn=False
            )
        }
    )
    
    print(f"Config manager patterns: {config_manager.get_all_patterns()}")
    
    # Define fuse patterns
    fuse_patterns = [
        ["conv", "bn", "relu"],
        ["linear", "bn", "relu"]
    ]
    
    # Fuse with config
    fused_model = fuse_modules_unified(
        model, 
        fuse_patterns, 
        is_trace=False, 
        config_manager=config_manager
    )
    
    print(f"Fused model has {sum(p.numel() for p in fused_model.parameters())} parameters")
    print(model)
    print("Fusion completed with programmatic config!\n")


def demo_yaml_config():
    """Demo using YAML configuration file."""
    print("=== Demo 2: YAML Configuration ===")
    
    # Create model
    model = SimpleModel()
    print(f"Original model has {sum(p.numel() for p in model.parameters())} parameters")
    
    try:
        # Load config from YAML
        config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")
        print(f"Loaded config from YAML with patterns: {config_manager.get_all_patterns()}")
        
        # Define fuse patterns
        fuse_patterns = [
            ["conv", "bn", "relu"],
            ["linear", "bn", "relu"]
        ]
        
        # Fuse with config
        fused_model = fuse_modules_unified(
            model, 
            fuse_patterns, 
            is_trace=False, 
            config_manager=config_manager
        )
        
        print(f"Fused model has {sum(p.numel() for p in fused_model.parameters())} parameters")
        print("Fusion completed with YAML config!\n")
        print(model)

        x = torch.rand(2,3,224,224)
        out = model(x)
        print(out.shape)
        
    except FileNotFoundError:
        print("YAML config file not found. Skipping YAML demo.\n")


def demo_default_config():
    """Demo using default configuration."""
    print("=== Demo 3: Default Configuration ===")
    
    # Create model
    model = SimpleModel()
    print(f"Original model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define fuse patterns
    fuse_patterns = [
        ["conv", "bn", "relu"],
        ["linear", "bn", "relu"]
    ]
    
    # Fuse without config (uses default)
    fused_model = fuse_modules_unified(
        model, 
        fuse_patterns, 
        is_trace=False
    )
    
    print(f"Fused model has {sum(p.numel() for p in fused_model.parameters())} parameters")
    print("Fusion completed with default config!\n")


def demo_layer_specific_config():
    """Demo showing how to add layer-specific configs dynamically."""
    print("=== Demo 4: Dynamic Layer-Specific Configuration ===")
    
    # Create model
    model = SimpleModel()
    print(f"Original model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create config manager
    config_manager = FuseConfigManager()
    
    # Add specific config for a particular layer
    config_manager.add_layer_config(
        "backbone_conv1", 
        FuseConfig(
            observer_w_name="MinMaxObserver",
            quantizer_w_name="UniformQuantizer",
            observer_a_name="MinMaxObserver",
            quantizer_a_name="UniformQuantizer",
            w_symmetric=False,
            a_symmetric=True,
            is_fuse_bn=True
        )
    )
    
    print(f"Added specific config for 'backbone_conv1'")
    print(f"Config manager patterns: {config_manager.get_all_patterns()}")
    
    # Define fuse patterns
    fuse_patterns = [
        ["conv", "bn", "relu"],
        ["linear", "bn", "relu"]
    ]
    
    # Fuse with config
    fused_model = fuse_modules_unified(
        model, 
        fuse_patterns, 
        is_trace=False, 
        config_manager=config_manager
    )
    
    print(f"Fused model has {sum(p.numel() for p in fused_model.parameters())} parameters")
    print("Fusion completed with dynamic layer-specific config!\n")

def test_quantization_utils():
    print("=== Test Quantization Utils ===")
    # Tạo model và fuse như demo_yaml_config
    model = SimpleModel()
    fuse_patterns = [
        ["conv", "bn", "relu"],
        ["linear", "bn", "relu"]
    ]
    config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")
    print(f"Loaded config from YAML with patterns: {config_manager.get_all_patterns()}")

    # Giả sử bạn có config_manager, hoặc dùng None để dùng mặc định
    fused_model = fuse_modules_unified(
        model, 
        fuse_patterns, 
        is_trace=False,
        config_manager=config_manager
    )

    print(fused_model)
    
    # Tạo dataloader giả lập
    dummy_data = torch.randn(4, 3, 224, 224)
    dataloader = [(dummy_data,)]  # Đơn giản hóa cho test

    print("qparam of fused module before:", fused_model.backbone_conv1.weight_quantizer.scale)
    # Calibration
    # calibrate_qat_model(fused_model, dataloader)
    # print("Calibration done.")

    # print("qparam of fused module after:", fused_model.backbone_conv1.weight_quantizer.scale)
    # # Bật learning scale cho toàn bộ model
    # activate_learning_qparam(fused_model)
    # print("Learning scale activated for all layers.")
    # print("qparam of fused module after init:", fused_model.backbone_conv1.activation_quantizer.scale)
    # print("--")
    # print("required_grad of fused module after:", fused_model.backbone_conv1.weight_quantizer.scale.requires_grad)



    # In trạng thái một số layer để xác nhận
    for name, module in fused_model.named_modules():
        if hasattr(module, "weight_quantizer"):
            print(f"{name}: is_learning_scale={module.weight_quantizer.is_learning_scale}, is_quantize={module.weight_quantizer.is_quantize}")

# Thêm vào cuối file hoặc gọi trong main()

def main():
    # set_seed(123)
    """Run all demos."""
    print("Fuse Configuration System Demo")
    print("=" * 50)
    
    # Run all demos
    # demo_programmatic_config()
    test_quantization_utils()
    # demo_default_config()
    # demo_layer_specific_config()
    
    print("All demos completed!")


if __name__ == "__main__":
    main() 