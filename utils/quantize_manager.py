import torch


def calibrate_qat_model(model, dataloader, data_calib, device=None):
    """
    Calibrate a quantization-aware training (QAT) model.
    
    This function:
    1. Enables calibration mode for all quantizers
    2. Runs forward passes to collect statistics
    3. Uses these statistics to compute quantization parameters
    
    Args:
        model (nn.Module): QAT model to calibrate
        dataloader: DataLoader providing calibration data
        data_calib: Function to run calibration forward passes
        device: Device to run calibration on
    """
    # Enable calibration mode for all quantizers
    for module in model.modules():
        if hasattr(module, "weight_quantizer"):
            module.weight_quantizer.is_observer_qparam = True
            module.weight_quantizer.is_learning_scale = False
            module.weight_quantizer.is_quantize = False
        if hasattr(module, "activation_quantizer"):
            module.activation_quantizer.is_observer_qparam = True
            module.activation_quantizer.is_learning_scale = False
            module.activation_quantizer.is_quantize = False
    model.eval()
    data_calib(model, dataloader, device)
    # After calibration, scale/zero_point have been collected


def activate_learning_qparam(model, layer_names=None, use_init=True, active=True):
    """
    Enable or disable learnable quantization parameters for specified layers.
    
    Args:
        model (nn.Module): QAT model to modify
        layer_names (list[str], optional): Names of layers to modify.
            If None, applies to all layers.
        use_init (bool): Whether to initialize scale factors
            using collected statistics
        active (bool): Whether to enable (True) or disable (False)
            learnable parameters
            
    For each targeted layer:
    1. Sets learning mode for quantization parameters
    2. Optionally initializes scale factors
    3. Makes parameters learnable if activating
    """
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.weight_quantizer.is_learning_scale = active
                if use_init:
                    module.weight_quantizer.init_scaling_factor_for_learning()
                if active:
                    module.weight_quantizer.make_learn_qparameter()
        if hasattr(module, "activation_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.activation_quantizer.is_learning_scale = active
                if use_init:
                    module.activation_quantizer.init_scaling_factor_for_learning()
                if active:
                    module.activation_quantizer.make_learn_qparameter()


def deactivate_learning_qparam(model, layer_names=None):
    """
    Disable learnable quantization parameters for specified layers.
    
    A convenience wrapper around activate_learning_qparam with active=False.
    
    Args:
        model (nn.Module): QAT model to modify
        layer_names (list[str], optional): Names of layers to modify.
            If None, applies to all layers.
    """
    activate_learning_qparam(model, layer_names=layer_names, active=False)


def activate_quantizer(model, layer_names=None, active=True):
    """
    Enable or disable quantization for specified layers.
    
    When enabled, the layer will actually perform quantization during
    forward passes. When disabled, it will pass through values unchanged.
    
    Args:
        model (nn.Module): QAT model to modify
        layer_names (list[str], optional): Names of layers to modify.
            If None, applies to all layers.
        active (bool): Whether to enable (True) or disable (False)
            quantization
    """
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.weight_quantizer.is_quantize = active
        if hasattr(module, "activation_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.activation_quantizer.is_quantize = active


def deactivate_quantizer(model, layer_names=None):
    """
    Disable quantization for specified layers.
    
    A convenience wrapper around activate_quantizer with active=False.
    
    Args:
        model (nn.Module): QAT model to modify
        layer_names (list[str], optional): Names of layers to modify.
            If None, applies to all layers.
    """
    activate_quantizer(model, layer_names=layer_names, active=False)

