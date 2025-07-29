import torch


def calibrate_qat_model(model, dataloader, data_calib, device=None):
    """
    Set calibration states for all quantizers, then run forward pass for observers to collect scale/zero_point.
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
    Enable (or disable) learning scale for all layers or specific layers.
    - model: QAT model
    - layer_names: list of layer names (str) or None to apply to all
    - active: True to enable, False to disable
    """
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.weight_quantizer.is_learning_scale = active
                if use_init == True:
                    module.weight_quantizer.init_scaling_factor_for_learning()
                if active == True:
                    module.weight_quantizer.make_learn_qparameter()
        if hasattr(module, "activation_quantizer"):
            if (layer_names is None) or (name in layer_names):
                module.activation_quantizer.is_learning_scale = active
                if use_init == True:
                    module.activation_quantizer.init_scaling_factor_for_learning()
                if active == True:
                    module.activation_quantizer.make_learn_qparameter()

def deactivate_learning_qparam(model, layer_names=None):
    """
    Disable learning scale for all layers or specific layers.
    """
    activate_learning_qparam(model, layer_names=layer_names, active=False)


def activate_quantizer(model, layer_names=None, active=True):
    """
    Enable (or disable) quantization for all layers or specific layers.
    - model: QAT model
    - layer_names: list of layer names (str) or None to apply to all
    - active: True to enable, False to disable
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
    Disable quantization for all layers or specific layers.
    """
    activate_quantizer(model, layer_names=layer_names, active=False)

