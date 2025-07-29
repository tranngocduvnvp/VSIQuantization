import re
import yaml
from typing import Dict, Optional, Union


class FuseConfig:
    """
    Configuration class for layer fusion and quantization parameters.
    
    This class holds configuration parameters for both weights and activations,
    including observer types, quantizer types, symmetry settings, and bit widths.
    
    Attributes:
        observer_w_name (str): Name of the observer for weights
        quantizer_w_name (str): Name of the quantizer for weights
        observer_a_name (str): Name of the observer for activations
        quantizer_a_name (str): Name of the quantizer for activations
        w_symmetric (bool): Whether to use symmetric quantization for weights
        a_symmetric (bool): Whether to use symmetric quantization for activations
        is_fuse_bn (bool): Whether to fuse batch normalization layers
        bits_w (int): Number of bits for weight quantization
        bits_a (int): Number of bits for activation quantization
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
    Manager class for handling layer-specific fusion and quantization configurations.
    
    This class manages multiple FuseConfig instances for different layers in a neural network,
    supporting pattern matching to apply specific configurations to groups of layers.
    
    Attributes:
        default_config (FuseConfig): Default configuration used when no specific match is found
        layer_configs (Dict[str, FuseConfig]): Mapping of layer patterns to their configurations
    """
    def __init__(self, default_config: Optional[FuseConfig] = None):
        self.default_config = default_config or FuseConfig()
        self.layer_configs: Dict[str, FuseConfig] = {}
    
    def add_layer_config(self, layer_pattern: str, config: FuseConfig):
        """
        Add a configuration for layers matching a specific pattern.
        
        Args:
            layer_pattern (str): Regular expression pattern to match layer names.
                               Falls back to substring matching if regex is invalid.
            config (FuseConfig): Configuration to apply to matching layers
        """
        self.layer_configs[layer_pattern] = config
    
    def get_config_for_layer(self, layer_name: str) -> FuseConfig:
        """
        Retrieve the configuration for a specific layer.
        
        Searches through registered patterns to find a matching configuration.
        Returns the default configuration if no pattern matches.
        
        Args:
            layer_name (str): Full name/path of the layer in the model
            
        Returns:
            FuseConfig: Configuration to use for the specified layer
        """
        for pattern, config in self.layer_configs.items():
            if self._match_pattern(layer_name, pattern):
                return config
        return self.default_config
    
    def _match_pattern(self, layer_name: str, pattern: str) -> bool:
        """
        Check if a layer name matches a pattern using regex or substring matching.
        
        First attempts to use regex matching, falls back to substring matching
        if the pattern is not a valid regex expression.
        
        Args:
            layer_name (str): Full name/path of the layer to check
            pattern (str): Pattern to match against (regex or substring)
            
        Returns:
            bool: True if the layer name matches the pattern, False otherwise
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
        """
        Set the default configuration used when no pattern matches.
        
        Args:
            config (FuseConfig): New default configuration
        """
        self.default_config = config
    
    def clear_layer_configs(self):
        """Remove all layer-specific configurations, leaving only the default config."""
        self.layer_configs.clear()
    
    def get_all_patterns(self) -> list:
        """
        Get all registered layer patterns.
        
        Returns:
            list: List of all pattern strings used for layer matching
        """
        return list(self.layer_configs.keys())
    
    def __repr__(self):
        return (f"FuseConfigManager(default={self.default_config}, "
                f"patterns={list(self.layer_configs.keys())})")


def load_fuse_config_from_yaml(yaml_path: str) -> FuseConfigManager:
    """
    Load fusion and quantization configurations from a YAML file.
    
    The YAML file should contain two main sections:
    1. 'default': Default configuration for all layers
    2. 'layers': Pattern-specific configurations for layer groups
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        FuseConfigManager: Manager instance with loaded configurations
        
    Raises:
        FileNotFoundError: If the specified YAML file doesn't exist
        ValueError: If the YAML file has invalid format
        
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
    Create a FuseConfigManager programmatically with custom configurations.
    
    This is an alternative to loading configurations from YAML, allowing
    direct creation of a config manager in code.
    
    Args:
        default_config (Optional[FuseConfig]): Default configuration for all layers
        layer_configs (Optional[Dict[str, Union[FuseConfig, Dict]]]): 
            Dictionary mapping layer patterns to their configurations.
            Values can be either FuseConfig objects or dictionaries of parameters.
    
    Returns:
        FuseConfigManager: Configured manager instance
        
    Raises:
        ValueError: If a layer config is neither a FuseConfig nor a dict
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