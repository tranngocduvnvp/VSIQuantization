import torch


def calibrate_qat_model(model, dataloader, data_calib, device=None):
    """
    Đặt các trạng thái calibration cho tất cả quantizer, sau đó chạy forward để observer thu thập scale/zero_point.
    """
    # Bật chế độ calibration cho tất cả quantizer
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
    # Sau calibration, scale/zero_point đã được thu thập


def activate_learning_qparam(model, layer_names=None, use_init=True, active=True):
    """
    Bật (hoặc tắt) learning scale cho tất cả các layer hoặc một số layer chỉ định.
    - model: mô hình QAT
    - layer_names: list tên lớp (str) hoặc None để áp dụng cho tất cả
    - active: True để bật, False để tắt
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
    Tắt learning scale cho tất cả các layer hoặc một số layer chỉ định.
    """
    activate_learning_qparam(model, layer_names=layer_names, active=False)


def activate_quantizer(model, layer_names=None, active=True):
    """
    Bật (hoặc tắt) quantization cho tất cả các layer hoặc một số layer chỉ định.
    - model: mô hình QAT
    - layer_names: list tên lớp (str) hoặc None để áp dụng cho tất cả
    - active: True để bật, False để tắt
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
    Tắt quantization cho tất cả các layer hoặc một số layer chỉ định.
    """
    activate_quantizer(model, layer_names=layer_names, active=False)

