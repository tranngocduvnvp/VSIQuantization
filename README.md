# Fuse Configuration System

Hệ thống cấu hình linh hoạt cho việc fuse các layer trong model quantization.

## Tổng quan

Hệ thống này cho phép bạn định nghĩa các tham số quantization và observer khác nhau cho từng layer hoặc nhóm layer, thay vì sử dụng các giá trị hardcode như trước đây.

## Các thành phần chính

### 1. FuseConfig
Class chứa các tham số cấu hình cho một layer:
- `observer_w_name`: Tên observer cho weights
- `quantizer_w_name`: Tên quantizer cho weights  
- `observer_a_name`: Tên observer cho activations
- `quantizer_a_name`: Tên quantizer cho activations
- `w_symmetric`: Có sử dụng symmetric quantization cho weights không
- `a_symmetric`: Có sử dụng symmetric quantization cho activations không
- `is_fuse_bn`: Có fuse batch normalization không

### 2. FuseConfigManager
Class quản lý các cấu hình cho nhiều layer:
- Hỗ trợ pattern matching (regex) để áp dụng config cho nhóm layer
- Có config mặc định cho các layer không match
- Cho phép thêm/xóa config động

### 3. Các hàm tiện ích
- `load_fuse_config_from_yaml()`: Load config từ file YAML
- `create_fuse_config_manager()`: Tạo config manager từ code

## Cách sử dụng

### 1. Sử dụng config mặc định

```python
from modules.fuse import fuse_modules_unified

# Fuse với config mặc định
fused_model = fuse_modules_unified(model, fuse_patterns)
```

### 2. Tạo config programmatically

```python
from modules.fuse_config import FuseConfig, create_fuse_config_manager
from modules.fuse import fuse_modules_unified

# Tạo config manager với các cấu hình tùy chỉnh
config_manager = create_fuse_config_manager(
    default_config=FuseConfig(
        observer_w_name="MinMaxObserver",
        quantizer_w_name="UniformQuantizer",
        w_symmetric=True,
        a_symmetric=True,
        is_fuse_bn=True
    ),
    layer_configs={
        "backbone.*conv": FuseConfig(
            observer_w_name="LSQObserver",
            quantizer_w_name="LSQQuantizer",
            w_symmetric=False,
            a_symmetric=True,
            is_fuse_bn=True
        ),
        "head.*linear": FuseConfig(
            observer_w_name="MinMaxObserver", 
            quantizer_w_name="UniformQuantizer",
            w_symmetric=True,
            a_symmetric=True,
            is_fuse_bn=False
        )
    }
)

# Fuse với config
fused_model = fuse_modules_unified(
    model, 
    fuse_patterns, 
    config_manager=config_manager
)
```

### 3. Load config từ file YAML

Tạo file `configs/fuse_config.yaml`:

```yaml
# Config mặc định
default:
  observer_w_name: "MinMaxObserver"
  quantizer_w_name: "UniformQuantizer"
  observer_a_name: "MinMaxObserver"
  quantizer_a_name: "UniformQuantizer"
  w_symmetric: true
  a_symmetric: true
  is_fuse_bn: true

# Config cho từng nhóm layer
layers:
  "backbone.*conv":
    observer_w_name: "LSQObserver"
    quantizer_w_name: "LSQQuantizer"
    w_symmetric: false
    a_symmetric: true
    is_fuse_bn: true
  
  "head.*linear":
    observer_w_name: "MinMaxObserver"
    quantizer_w_name: "UniformQuantizer"
    w_symmetric: true
    a_symmetric: true
    is_fuse_bn: false
```

Load và sử dụng:

```python
from modules.fuse_config import load_fuse_config_from_yaml
from modules.fuse import fuse_modules_unified

# Load config từ YAML
config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")

# Fuse với config
fused_model = fuse_modules_unified(
    model, 
    fuse_patterns, 
    config_manager=config_manager
)
```

### 4. Thêm config động

```python
from modules.fuse_config import FuseConfigManager, FuseConfig

# Tạo config manager
config_manager = FuseConfigManager()

# Thêm config cho layer cụ thể
config_manager.add_layer_config(
    "features.0",
    FuseConfig(
        observer_w_name="PerChannelMinMaxObserver",
        quantizer_w_name="PerChannelUniformQuantizer",
        w_symmetric=False,
        a_symmetric=True,
        is_fuse_bn=True
    )
)

# Fuse với config
fused_model = fuse_modules_unified(
    model, 
    fuse_patterns, 
    config_manager=config_manager
)
```

## Pattern Matching

Hệ thống hỗ trợ pattern matching để áp dụng config cho nhóm layer:

- `"backbone.*conv"`: Tất cả conv layer trong backbone
- `"head.*linear"`: Tất cả linear layer trong head  
- `".*stem.*"`: Tất cả layer chứa "stem"
- `"features.0"`: Layer cụ thể có tên "features.0"

Pattern matching sử dụng regex, nếu regex không hợp lệ sẽ fallback về substring matching.

## API Reference

### FuseConfig

```python
class FuseConfig:
    def __init__(
        self,
        observer_w_name: str = "MinMaxObserver",
        quantizer_w_name: str = "UniformQuantizer",
        observer_a_name: str = "MinMaxObserver", 
        quantizer_a_name: str = "UniformQuantizer",
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        is_fuse_bn: bool = True
    )
```

### FuseConfigManager

```python
class FuseConfigManager:
    def __init__(self, default_config: Optional[FuseConfig] = None)
    def add_layer_config(self, layer_pattern: str, config: FuseConfig)
    def get_config_for_layer(self, layer_name: str) -> FuseConfig
    def set_default_config(self, config: FuseConfig)
    def clear_layer_configs(self)
    def get_all_patterns(self) -> list
```

### Hàm tiện ích

```python
def load_fuse_config_from_yaml(yaml_path: str) -> FuseConfigManager
def create_fuse_config_manager(
    default_config: Optional[FuseConfig] = None,
    layer_configs: Optional[Dict[str, Union[FuseConfig, Dict]]] = None
) -> FuseConfigManager
```

## Ví dụ hoàn chỉnh

Xem file `examples/fuse_config_demo.py` để có ví dụ chi tiết về cách sử dụng tất cả các tính năng.

## Lưu ý

1. Config được áp dụng cho layer đầu tiên trong pattern fuse
2. Nếu không tìm thấy config cho layer, sẽ sử dụng config mặc định
3. Hệ thống tương thích ngược với code cũ (không cần config_manager)
4. Hỗ trợ cả tracing và non-tracing fusion 