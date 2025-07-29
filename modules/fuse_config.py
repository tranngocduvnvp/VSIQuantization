import re
import yaml
from typing import Dict, Optional, Union


class FuseConfig:
    """
    Configuration class for fused layer parameters.
    """
    def __init__(
        self,
        observer_w_name: str = "MinMaxObserver",
        quantizer_w_name: str = "UniformQuantizer",
        observer_a_name: str = "MinMaxObserver",
        quantizer_a_name: str = "UniformQuantizer",
        w_symmetric: bool = True,
        a_symmetric: bool = True,
        is_fuse_bn: bool = True,
        bits_w: int = 8,
        bits_a: int = 8
    ):
        self.observer_w_name = observer_w_name
        self.quantizer_w_name = quantizer_w_name
        self.observer_a_name = observer_a_name
        self.quantizer_a_name = quantizer_a_name
        self.w_symmetric = w_symmetric
        self.a_symmetric = a_symmetric
        self.is_fuse_bn = is_fuse_bn
        self.bits_w = bits_w
        self.bits_a = bits_a
    
    def __repr__(self):
        return (f"FuseConfig(observer_w='{self.observer_w_name}', "
                f"quantizer_w='{self.quantizer_w_name}', "
                f"observer_a='{self.observer_a_name}', "
                f"quantizer_a='{self.quantizer_a_name}', "
                f"w_symmetric={self.w_symmetric}, "
                f"a_symmetric={self.a_symmetric}, "
                f"is_fuse_bn={self.is_fuse_bn}, "
                f"bits_w={self.bits_w}, bits_a={self.bits_a})")


class FuseConfigManager:
    """
    Manager class for handling layer-specific fuse configurations.
    """
    def __init__(self, default_config: Optional[FuseConfig] = None):
        self.default_config = default_config or FuseConfig()
        self.layer_configs: Dict[str, FuseConfig] = {}
    
    def add_layer_config(self, layer_pattern: str, config: FuseConfig):
        """
        Add configuration for layers matching the pattern.
        
        Args:
            layer_pattern: Pattern to match layer names (supports regex)
            config: FuseConfig instance for matching layers
        """
        self.layer_configs[layer_pattern] = config
    
    def get_config_for_layer(self, layer_name: str) -> FuseConfig:
        """
        Get configuration for a specific layer.
        
        Args:
            layer_name: Full name of the layer
            
        Returns:
            FuseConfig instance for the layer, or default config if no match
        """
        for pattern, config in self.layer_configs.items():
            if self._match_pattern(layer_name, pattern):
                return config
        return self.default_config
    
    def _match_pattern(self, layer_name: str, pattern: str) -> bool:
        """
        Check if layer_name matches the pattern.
        
        Args:
            layer_name: Full name of the layer
            pattern: Pattern to match against (supports regex)
            
        Returns:
            True if layer_name matches pattern, False otherwise
        """
        try:
            # Try regex matching first
            if re.search(pattern, layer_name):
                return True
        except re.error:
            # If regex fails, try simple substring matching
            if pattern in layer_name:
                return True
        return False
    
    def set_default_config(self, config: FuseConfig):
        """Set the default configuration."""
        self.default_config = config
    
    def clear_layer_configs(self):
        """Clear all layer-specific configurations."""
        self.layer_configs.clear()
    
    def get_all_patterns(self) -> list:
        """Get all registered layer patterns."""
        return list(self.layer_configs.keys())
    
    def __repr__(self):
        return (f"FuseConfigManager(default={self.default_config}, "
                f"patterns={list(self.layer_configs.keys())})")


def load_fuse_config_from_yaml(yaml_path: str) -> FuseConfigManager:
    """
    Load fuse configuration from YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        FuseConfigManager instance with loaded configurations
        
    Example YAML structure:
    default:
      observer_w_name: "MinMaxObserver"
      quantizer_w_name: "UniformQuantizer"
      observer_a_name: "MinMaxObserver"
      quantizer_a_name: "UniformQuantizer"
      w_symmetric: true
      a_symmetric: true
      is_fuse_bn: true
    
    layers:
      "backbone.*conv":
        observer_w_name: "LSQObserver"
        quantizer_w_name: "LSQQuantizer"
        w_symmetric: false
        a_symmetric: false
        is_fuse_bn: true
      
      "head.*linear":
        observer_w_name: "MinMaxObserver"
        quantizer_w_name: "UniformQuantizer"
        w_symmetric: true
        a_symmetric: true
        is_fuse_bn: false
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")
    
    manager = FuseConfigManager()
    
    # Set default config
    if 'default' in config_data:
        default_data = config_data['default']
        manager.default_config = FuseConfig(**default_data)
    
    # Add layer-specific configs
    if 'layers' in config_data:
        for layer_pattern, layer_config in config_data['layers'].items():
            config = FuseConfig(**layer_config)
            manager.add_layer_config(layer_pattern, config)
    
    return manager


def create_fuse_config_manager(
    default_config: Optional[FuseConfig] = None,
    layer_configs: Optional[Dict[str, Union[FuseConfig, Dict]]] = None
) -> FuseConfigManager:
    """
    Create a FuseConfigManager with optional default and layer-specific configs.
    
    Args:
        default_config: Default configuration for all layers
        layer_configs: Dictionary mapping layer patterns to configs
                      Can be FuseConfig objects or dictionaries
    
    Returns:
        FuseConfigManager instance
    """
    manager = FuseConfigManager(default_config)
    
    if layer_configs:
        for pattern, config in layer_configs.items():
            if isinstance(config, dict):
                config = FuseConfig(**config)
            elif not isinstance(config, FuseConfig):
                raise ValueError(f"Config for pattern '{pattern}' must be FuseConfig or dict")
            manager.add_layer_config(pattern, config)
    
    return manager 