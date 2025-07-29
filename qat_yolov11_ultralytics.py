from ultralytics import YOLO
from yolov8_qat import data_calib
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
    deactivate_quantizer,
)
from utils.estimate_bn import reestimate_BN_stats, compute_scale



import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    YAML,
    callbacks,
    checks,
)
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from dataset.dataset import Dataset
from torch.utils import data
import glob
import os
# Path to data configuration and model files
data_yaml = 'configs/voc.yaml'  # Path to the voc.yaml file you just created

import yaml

import matplotlib.pyplot as plt
# Callback function to plot realtime histogram after each batch
fig, ax = plt.subplots(figsize=(8, 4))
plt.ion()
def plot_weight_hist_realtime(weight_tensor, iteration, fig, ax):
    ax.clear()
    weight_int = torch.round(weight_tensor).to(torch.int).cpu().numpy().flatten()
    ax.hist(weight_int, bins=50, color='blue', alpha=0.7)
    ax.set_title(f'Quantized Weight Histogram - Iter {iteration}')
    ax.set_xlabel('Quantized Weight Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# Callback for Ultralytics
# This function will be called after each batch
# You can change 'model.model[0].conv.weight' to the layer you want to monitor



config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")
fuse_patterns = [
        ["conv", "bn", "relu"],
        # ["linear", "bn", "relu"],
        ["conv", "bn"],
        # ["conv", "relu"],
        # ["linear", "bn"],
        # ["linear", "relu"],
        # ["conv"]
    ]



class YOLOCustom(YOLO):
    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        with open("/data/dutn1/VSIQuantization/nets/args_voc.yaml", errors='ignore') as f:
            params = yaml.safe_load(f)

        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            self.load(kwargs["pretrained"])  # load pretrained weights if provided
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
            self.model = fuse_modules_unified(
                            self.model, 
                            fuse_patterns, 
                            is_trace=False, 
                            config_manager=config_manager
                        )
            dataset_calib = Dataset(glob.glob(os.path.join("/data/dutn1/VSIQuantization/datasets/VOC", 'images/train2012/*')), 320, params, False)
            self.trainer.calib_loader = data.DataLoader(dataset_calib,320,
                                    num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

            calibrate_qat_model(self.model, self.trainer.calib_loader, data_calib, "cuda:0")
            activate_learning_qparam(self.model, use_init=True)
            activate_quantizer(self.model)
            with open("./model_structure.txt", "w") as f:
                f.write(str(self.model))
            

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics
    

model = YOLOCustom("/data/dutn1/VSIQuantization/runs/train/yolov11-voc/weights/base_voc_yolov11.pt")
# model = YOLO("/data/dutn1/VSIQuantization/runs/train/yolov11-voc/weights/base_voc_yolov11.pt")
def on_train_batch_end(trainer):
    with open("./scale_param.txt", "w") as f2:
        for n, m in trainer.model.named_parameters():
            if n.endswith("scale"):
                str_val = ",".join(str(v) for v in m.view(-1).tolist())
                try:
                    str_val_grad = ",".join(str(v) for v in m.grad.view(-1).tolist())
                except:
                    str_val_grad = "None"
                f2.write(f"{n}: [{str_val}],  grad: [{str_val_grad}]\n")
            


def on_train_epoch_end(trainer):
    compute_scale(trainer.model, trainer.calib_loader, 50)

model.add_callback("on_train_batch_end", on_train_batch_end)
# model.add_callback("on_train_epoch_end", on_train_epoch_end)
# Training
model.train(
    data=data_yaml,
    epochs=100,
    imgsz=320,
    batch=64,
    device=0,  # or 'cpu' if no GPU is available
    workers=4,
    project='runs/train',
    name='yolov11-voc',
    lr0=0.1,
    exist_ok=True
)

# model.val(
#     data=data_yaml,
#     imgsz=320,
#     batch=128,
#     device=0,  # or 'cpu' if no GPU is available
#     workers=4,
#     plots=True,
#     name='yolov11-voc',
# )