# VSIQuantization

A flexible quantization-aware training (QAT) framework for deep neural networks with support for layer fusion and learnable quantization parameters.

## Features

- Multiple quantization schemes:
  - Uniform quantization
  - Learned Step-size Quantization (LSQ)
  - Support for symmetric and asymmetric quantization
  
- Layer fusion capabilities:
  - Conv + BatchNorm + ReLU/SiLU
  - Conv + BatchNorm
  - Conv + ReLU/SiLU
  - Linear + BatchNorm + ReLU/SiLU
  - Linear + BatchNorm
  - Linear + ReLU/SiLU

- Flexible configuration system:
  - Per-layer quantization settings
  - Pattern-based layer matching
  - YAML configuration support
  
- Advanced calibration:
  - Batch normalization statistics re-estimation
  - Gradient scaling optimization
  - Statistical parameter initialization

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/VSIQuantization.git
cd VSIQuantization

# Install dependencies
pip install torch torchvision
```

## Usage

### Basic Usage

```python
from modules.fuse import fuse_modules_unified

# Fuse layers with default configuration
fused_model = fuse_modules_unified(model, fuse_patterns)
```

### Custom Configuration

```python
from modules.fuse_config import FuseConfig, create_fuse_config_manager

# Create custom configuration
config_manager = create_fuse_config_manager(
    default_config=FuseConfig(
        observer_w_name="MinMaxObserver",
        quantizer_w_name="UniformQuantizer",
        w_symmetric=True,
        a_symmetric=True,
        is_fuse_bn=True,
        bits_w=8,
        bits_a=8
    ),
    layer_configs={
        "backbone.*conv": FuseConfig(
            observer_w_name="LSQObserver",
            quantizer_w_name="LSQQuantizer",
            w_symmetric=False,
            a_symmetric=True,
            is_fuse_bn=True
        )
    }
)

# Fuse with custom config
fused_model = fuse_modules_unified(
    model, 
    fuse_patterns,
    config_manager=config_manager
)
```

### YAML Configuration

Create `configs/fuse_config.yaml`:

```yaml
default:
  observer_w_name: "MinMaxObserver"
  quantizer_w_name: "UniformQuantizer"
  observer_a_name: "MinMaxObserver"
  quantizer_a_name: "UniformQuantizer"
  w_symmetric: true
  a_symmetric: true
  is_fuse_bn: true
  bits_w: 8
  bits_a: 8

layers:
  "backbone.*conv":
    observer_w_name: "LSQObserver"
    quantizer_w_name: "LSQQuantizer"
    w_symmetric: false
    a_symmetric: true
    is_fuse_bn: true
```

Load and use the configuration:

```python
from modules.fuse_config import load_fuse_config_from_yaml

config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")
fused_model = fuse_modules_unified(model, fuse_patterns, config_manager=config_manager)
```

## Components

### Quantizers

- `UniformQuantizer`: Basic uniform quantization
- `LSQQuantizer`: Learned Step-size Quantization with learnable scale/zero-point

### Observers

- `MinMaxObserver`: Tracks min/max values for scale computation
- `LSQObserver`: Observer for LSQ quantization

### Fused Layers

- `ConvBnReLU`: Fused Conv2d + BatchNorm2d + ReLU/SiLU
- `ConvBn`: Fused Conv2d + BatchNorm2d
- `ConvReLU`: Fused Conv2d + ReLU/SiLU
- `LinearBnReLU`: Fused Linear + BatchNorm1d + ReLU/SiLU
- `LinearBn`: Fused Linear + BatchNorm1d
- `LinearReLU`: Fused Linear + ReLU/SiLU

### Utilities

- Batch normalization statistics re-estimation
- Gradient scaling computation
- Model checkpoint management
- Pattern-based layer matching

## Advanced Features

### Batch Norm Re-estimation

```python
from utils.estimate_bn import reestimate_BN_stats

# Re-estimate BN statistics using calibration data
reestimate_BN_stats(model, data_loader, num_batches=50)
```

### Gradient Scale Optimization

```python
from utils.estimate_bn import compute_scale

# Compute optimal gradient scales
compute_scale(model, data_loader)
```

### Quantization Management

```python
from utils.quantize_manager import calibrate_qat_model, activate_learning_qparam

# Calibrate quantization parameters
calibrate_qat_model(model, data_loader, data_calib)

# Enable learnable parameters
activate_learning_qparam(model)
```

## Pattern Matching

The configuration system supports pattern matching for layer groups:

- `"backbone.*conv"`: All conv layers in backbone
- `"head.*linear"`: All linear layers in head
- `".*stem.*"`: All layers containing "stem"
- `"features.0"`: Specific layer named "features.0"

Pattern matching uses regex with fallback to substring matching.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 