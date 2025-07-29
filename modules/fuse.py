import torch
import torch.nn as nn
import torch.fx as fx
from modules.fused import ConvBnReLU, ConvBn,\
     ConvReLU, LinearBnReLU, LinearBn, LinearReLU, Conv, Linear
from modules.fuse_config import FuseConfigManager, FuseConfig

# Mapping of layer pattern tuples to their corresponding fused layer classes
PATTERN_TO_FUSED = {
    ("conv", "bn", "relu"): ConvBnReLU,
    ("conv", "bn"): ConvBn,
    ("conv", "relu"): ConvReLU,
    ("linear", "bn", "relu"): LinearBnReLU,
    ("linear", "bn"): LinearBn,
    ("linear", "relu"): LinearReLU,
    ("conv",): Conv,
    ("linear",): Linear,
}

# Mapping of PyTorch module types to their string representations
MODULE_TYPE_TO_STR = {
    nn.Conv2d: "conv",
    nn.Linear: "linear",
    nn.BatchNorm2d: "bn",
    nn.BatchNorm1d: "bn",
    nn.ReLU: "relu",
}

def get_module_type_str(module):
    """
    Get the string representation of a module's type.
    
    Args:
        module (nn.Module): The module to get the type string for
        
    Returns:
        str or None: The string representation of the module type,
                    or None if the type is not recognized
    """
    for t, s in MODULE_TYPE_TO_STR.items():
        if isinstance(module, t):
            return s
    return None

def _fuse_modules(model, fuse_patterns, config_manager=None):
    """
    Fuse model layers directly without using tracing (shallow fusion).
    
    This method performs layer fusion by directly manipulating the model's
    module hierarchy. It's more reliable but less flexible than tracing-based fusion.
    
    Args:
        model (nn.Module): The PyTorch model to fuse
        fuse_patterns (List[List[str]]): List of layer patterns to fuse (e.g. [["conv", "bn", "relu"]])
        config_manager (Optional[FuseConfigManager]): Manager providing fusion configurations
        
    Returns:
        nn.Module: The model with fused layers
        
    Note:
        This method is similar to torch.quantization.fuse_modules but adds support for:
        - Custom fusion patterns
        - Layer-specific quantization parameters
        - Flexible batch normalization fusion
    """
    if config_manager is None:
        config_manager = FuseConfigManager()  # Use default config
    
    for pattern in fuse_patterns:
        pattern_tuple = tuple(pattern)
        fused_class = PATTERN_TO_FUSED.get(pattern_tuple, None)
        if fused_class is None:
            continue
        modules_to_fuse = []
        for module_name, module in model.named_modules():
            # Skip if module can't contain child modules or is already fused
            if not hasattr(module, "_modules") or\
                isinstance(module,(ConvBnReLU, ConvBn, ConvReLU, Conv,\
                     LinearBnReLU, LinearBn, LinearReLU, Linear)) \
                    or len(module._modules) < len(pattern):
                continue

            child_names = list(module._modules.keys())
            i = 0
            while i <= len(child_names) - len(pattern):
                match = True
                for j, name in enumerate(pattern):
                    child = module._modules[child_names[i+j]]
                    # Check if each module in the sequence matches the pattern
                    if name == "conv" and not isinstance(child, nn.Conv2d):
                        match = False
                        break
                    if name == "linear" and not isinstance(child, nn.Linear):
                        match = False
                        break
                    if name == "bn" and not (isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d)):
                        match = False
                        break
                    if name == "relu" and not isinstance(child, (nn.ReLU, nn.SiLU)):
                        match = False
                        break
                if match:
                    modules_to_fuse.append((module, child_names[i:i+len(pattern)], fused_class))
                    i += len(pattern)
                else:
                    i += 1
        
        # Perform fusion for matched module sequences
        for module, names, fused_class in modules_to_fuse:
            fused_args = [module._modules[n] for n in names]
            
            # Get config for the first layer in the pattern
            first_layer_name = names[0]
            config = config_manager.get_config_for_layer(first_layer_name)
            
            # Add quantization parameters from config
            if fused_class in [ConvBnReLU, ConvBn, LinearBnReLU, LinearBn]:
                # These classes need is_fuse_bn parameter
                fused_args.extend([
                    config.observer_w_name,
                    config.quantizer_w_name,
                    config.observer_a_name,
                    config.quantizer_a_name,
                    config.w_symmetric,
                    config.a_symmetric,
                    config.is_fuse_bn,
                    config.bits_w,
                    config.bits_a
                ])
            else:
                # ConvReLU and LinearReLU don't need is_fuse_bn
                fused_args.extend([
                    config.observer_w_name,
                    config.quantizer_w_name,
                    config.observer_a_name,
                    config.quantizer_a_name,
                    config.w_symmetric,
                    config.a_symmetric,
                    config.bits_w,
                    config.bits_a
                ])
            
            # Create and insert fused module
            fused = fused_class(*fused_args)
            module._modules[names[0]] = fused
            # Replace other modules with Identity
            for n in names[1:]:
                module._modules[n] = nn.Identity()
    return model

def _fuse_modules_trace(model, fuse_patterns, config_manager=None):
    """
    Fuse model layers using torch.fx tracing.
    
    This method uses symbolic tracing to analyze the model's computation graph
    and perform fusion. It's more flexible than direct fusion but may fail for
    models with dynamic control flow.
    
    Args:
        model (nn.Module): The PyTorch model to fuse
        fuse_patterns (List[List[str]]): List of layer patterns to fuse (e.g. [["conv", "bn", "relu"]])
        config_manager (Optional[FuseConfigManager]): Manager providing fusion configurations
        
    Returns:
        nn.Module: The model with fused layers
        
    Note:
        Falls back to _fuse_modules if tracing fails
    """
    if config_manager is None:
        config_manager = FuseConfigManager()  # Use default config
        
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        print(f"Warning: symbolic_trace failed: {e}")
        print("Falling back to direct fuse method")
        return _fuse_modules(model, fuse_patterns, config_manager)

    modules = dict(gm.named_modules())
    nodes = list(gm.graph.nodes)
    i = 0
    while i < len(nodes):
        for pattern in fuse_patterns:
            if i + len(pattern) > len(nodes):
                continue
            match = True
            node_seq = nodes[i:i+len(pattern)]
            module_seq = []
            
            # Check if sequence of nodes matches the pattern
            for j, name in enumerate(pattern):
                node = node_seq[j]
                if node is None or node.op != 'call_module':
                    match = False
                    break
                mod = modules.get(node.target, None)
                if mod is None or get_module_type_str(mod) != name:
                    match = False
                    break
                module_seq.append(mod)
                
            if match:
                fused_class = PATTERN_TO_FUSED[tuple(pattern)]
                
                # Get config for the first layer in the pattern
                first_layer_name = node_seq[0].target
                config = config_manager.get_config_for_layer(first_layer_name)
                
                # Add quantization parameters from config
                if fused_class in [ConvBnReLU, ConvBn, LinearBnReLU, LinearBn]:
                    # These classes need is_fuse_bn parameter
                    module_seq.extend([
                        config.observer_w_name,
                        config.quantizer_w_name,
                        config.observer_a_name,
                        config.quantizer_a_name,
                        config.w_symmetric,
                        config.a_symmetric,
                        config.is_fuse_bn,
                        config.bits_w,
                        config.bits_a
                    ])
                else:
                    # ConvReLU and LinearReLU don't need is_fuse_bn
                    module_seq.extend([
                        config.observer_w_name,
                        config.quantizer_w_name,
                        config.observer_a_name,
                        config.quantizer_a_name,
                        config.w_symmetric,
                        config.a_symmetric,
                        config.bits_w,
                        config.bits_a
                    ])
                
                # Create and insert fused module
                fused = fused_class(*module_seq)
                parent_name = '.'.join(node_seq[0].target.split('.')[:-1])
                fused_name = node_seq[0].target.split('.')[-1]
                parent = model if parent_name == '' else modules[parent_name]
                setattr(parent, fused_name, fused)
                
                # Remove other modules and mark nodes as processed
                for n in node_seq[1:]:
                    delattr(parent, n.target.split('.')[-1])
                for j in range(1, len(pattern)):
                    nodes[i+j] = None
                i += len(pattern)
                break
        i += 1
    return model

def fuse_modules_unified(model, fuse_patterns, is_trace=False, config_manager=None):
    """
    Unified interface for model layer fusion.
    
    This function provides a single entry point for both tracing-based and
    direct fusion methods. The tracing method is more flexible but may fail
    for complex models, while the direct method is more reliable but less
    flexible.
    
    Args:
        model (nn.Module): The PyTorch model to fuse
        fuse_patterns (List[List[str]]): List of layer patterns to fuse
        is_trace (bool): Whether to use tracing-based fusion (True) or
                        direct fusion (False)
        config_manager (Optional[FuseConfigManager]): Manager providing
                                                    fusion configurations
        
    Returns:
        nn.Module: The model with fused layers
    """
    if is_trace:
        return _fuse_modules_trace(model, fuse_patterns, config_manager)
    else:
        return _fuse_modules(model, fuse_patterns, config_manager) 