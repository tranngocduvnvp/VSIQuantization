
import torch
import torch.nn as nn
from torch.utils import data
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
from dataset.dataset import Dataset
import glob
import os
from torch.utils.data import random_split
from nets.yolov8 import yolo_v8_n
from utils.util import *
import csv
import tqdm
import pandas as pd
import yaml
from argparse import ArgumentParser
import warnings
import mlflow
import mlflow.pytorch
import mlflow.models
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def data_calib(model, calib_loader, device):
    print("Calibration dataset .....")
    model.eval()
    model.to(device)
    for batch_i, (imgs, targets) in (enumerate(calib_loader)):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        model(imgs)

        if batch_i == 15:
            break
    model.train()

def train(args, params):
    setup_seed()
    setup_multi_processes()
    device = "cuda:0"
    qat_model = yolo_v8_n(num_classes=20)

    load_partial_checkpoint(qat_model, os.path.join(args.result_path, 'best_voc_checkpoint.pth'))

    

    config_manager = load_fuse_config_from_yaml("configs/fuse_config.yaml")
    fuse_patterns = [
            ["conv", "bn", "relu"],
            # ["linear", "bn", "relu"],
            # ["conv", "bn"],
            # ["conv", "relu"],
            # ["linear", "bn"],
            # ["linear", "relu"],
            # ["conv"]
        ]

    qat_model = fuse_modules_unified(
            qat_model, 
            fuse_patterns, 
            is_trace=False, 
            config_manager=config_manager
        )
    
    # print(qat_model)

    # exit()
    # # Load datasets
    dataset_calib = Dataset(glob.glob(os.path.join(args.dataset_path, 'images/train2012/*')), args.input_size, params, False)
    calib_loader = data.DataLoader(dataset_calib, args.batch_size,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    calibrate_qat_model(qat_model, calib_loader, data_calib, "cuda")
    activate_learning_qparam(qat_model, use_init=True)
    activate_quantizer(qat_model)

    # qat_model.head.box[0][2].quantize_inp = False
    # qat_model.head.box[0][2].quantize_out = False
    # qat_model.head.box[1][2].quantize_inp = False
    # qat_model.head.box[1][2].quantize_out = False
    # qat_model.head.box[2][2].quantize_inp = False
    # qat_model.head.box[2][2].quantize_out = False

    # qat_model.head.cls[0][2].quantize_inp = False
    # qat_model.head.cls[0][2].quantize_out = False
    # qat_model.head.cls[1][2].quantize_inp = False
    # qat_model.head.cls[1][2].quantize_out = False
    # qat_model.head.cls[2][2].quantize_inp = False
    # qat_model.head.cls[2][2].quantize_out = False


    with open("./model_structure.txt", "w") as f:
        f.write(str(qat_model))

    qat_model.to(device)

    print(qat_model.net.p1[0].conv.weight_quantizer.scale)
    print(qat_model.net.p1[0].conv.activation_quantizer.scale)
    # print(qat_model.net.p1[0].conv.conv_fuse.weight)
    # exit()
    

    filenames = []
    filenames = glob.glob(os.path.join(args.dataset_path, 'images/train2012/*')) +\
         glob.glob(os.path.join(args.dataset_path, 'images/train2007/*'))
    sampler = None
    dataset = Dataset(filenames, args.input_size, params, True)

    # # make subset for dev
    # total_size = len(dataset)
    # subset_size = int(0.01* total_size)
    # rest_size = total_size - subset_size
    # dataset, _ = random_split(dataset,[subset_size, rest_size], torch.Generator().manual_seed(42))
    # # the end -->


    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    if args.distributed:
        # DDP mode
        qat_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(qat_model)
        qat_model = torch.nn.parallel.DistributedDataParallel(module=qat_model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    best = 0
    num_steps = len(loader)
    criterion = ComputeLoss(qat_model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_steps), 100)

    optimizer = torch.optim.SGD(weight_decay(qat_model, params['weight_decay']),
                                params['lr0'], params['momentum'], nesterov=True)

    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)
    

    log_dict = {}
    for k, v in qat_model.named_parameters():
        if k.endswith("weight"):
            log_dict[k] = []

    with open(os.path.join(args.result_path, 'weights_qat/step_qat.csv'), 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch',
                                                   'box', 'cls',
                                                   'Recall', 'Precision', 'mAP@50', 'mAP'])
            writer.writeheader()
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'input_size': args.input_size,
                'learning_rate': params.get('lr0', None),
                'momentum': params.get('momentum', None),
                'weight_decay': params.get('weight_decay', None),
            })
            input_example = torch.zeros(1, 3, args.input_size, args.input_size).numpy()
            for epoch in range(args.epochs):
                qat_model.train()
                if args.distributed:
                    sampler.set_epoch(epoch)
                
                loader.dataset.mosaic = False 

                if epoch == 20:
                    qat_model = set_untrainable_layers(qat_model)

                p_bar = enumerate(loader)

                if args.local_rank == 0:
                    print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'box', 'cls'))
                if args.local_rank == 0:
                    p_bar = tqdm.tqdm(p_bar, total=num_steps)  # progress bar

                optimizer.zero_grad()
                avg_box_loss = AverageMeter()
                avg_cls_loss = AverageMeter()
                


                for i, (samples, targets) in p_bar:
                    samples = samples.cuda()
                    samples = samples.float()
                    samples = samples / 255.0

                    x = i + num_steps * epoch
                    # Warmup
                    if x <= num_warmup:
                        xp = [0, num_warmup]
                        fp = [1, 64 / (args.batch_size * args.world_size)]
                        accumulate = max(1, numpy.interp(x, xp, fp).round())
                        for j, y in enumerate(optimizer.param_groups):
                            if j == 0:
                                fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                            else:
                                fp = [0.0, y['initial_lr'] * lr(epoch)]
                            y['lr'] = numpy.interp(x, xp, fp)
                            if 'momentum' in y:
                                fp = [params['warmup_momentum'], params['momentum']]
                                y['momentum'] = numpy.interp(x, xp, fp)
                   

                    # Forward
                    outputs = qat_model(samples)
                    # print(outputs[0].shape)
                    # print(outputs[1].shape)
                    # print(outputs[2].shape)
                    loss_box, loss_cls = criterion(outputs, targets)
                    avg_box_loss.update(loss_box.item(), samples.size(0))
                    avg_cls_loss.update(loss_cls.item(), samples.size(0))

                    loss_box *= args.batch_size  # loss scaled by batch_size
                    loss_cls *= args.batch_size  # loss scaled by batch_size
                    loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                    loss_cls *= args.world_size  # gradient averaged between devices in DDP mode

                    # Backward
                    (loss_box + loss_cls).backward()

                    # Log scale mean of each layer to MLflow
                    # for n, m in qat_model.named_parameters():
                    #     if n.endswith("scale"):
                    #         scale_val = m.detach().mean().item()
                    #         metric_name = f"scale_{n.replace('.', '_')}"
                    #         mlflow.log_metric(metric_name, scale_val, step=x)

                    with open("./scale_param.txt", "w") as f2:
                        for n, m in qat_model.named_parameters():
                            if n.endswith("scale"):
                                str_val = ",".join(str(v) for v in m.view(-1).tolist())
                                try:
                                    str_val_grad = ",".join(str(v) for v in m.grad.view(-1).tolist())
                                except:
                                    str_val_grad = "None"
                                f2.write(f"{n}: [{str_val}],  grad: [{str_val_grad}]\n")
                            elif n.endswith("weight"):
                                log_dict[n].append(m.view(-1)[0].item())

                    if x % accumulate == 0:
                        clip_gradients(qat_model)  # clip gradients
                        optimizer.step()
                        optimizer.zero_grad()

                    # Log
                    if args.local_rank == 0:
                        memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                        s = ('%10s' * 2 + '%10.3g' * 2) % (f'{epoch + 1}/{args.epochs}', memory,
                                                           avg_box_loss.avg, avg_cls_loss.avg)
                        p_bar.set_description(s)
                    
                        frsave = pd.DataFrame(log_dict)
                        frsave.to_csv("./weight_fluc.csv", index=False)
                    
                    
                # Scheduler
                scheduler.step()



                if args.local_rank == 0:
                    # Convert model
                    # save = copy.deepcopy(model.module if args.distributed else model)
                    save = qat_model.module if args.distributed else qat_model
                    save.eval()
                   
                    last = test(args, params, save, device="cuda")
                    
                    # Log metrics for each epoch
                    mlflow.log_metrics({
                        'box_loss': avg_box_loss.avg,
                        'cls_loss': avg_cls_loss.avg,
                        'mAP': last[0],
                        'mAP_50': last[1],
                        'Recall': last[2],
                        'Precision': last[3]
                    }, step=epoch)
                    # Log model checkpoint for each epoch (optional, can log only the best)
                    torch.save(save.state_dict(), os.path.join(args.result_path, 'last_qat.pth'))
                    input_tensor = torch.from_numpy(input_example).to(device).float()
                    mlflow.pytorch.log_model(
                        save,
                        name="qat_model_epoch_{}".format(epoch+1),
                        input_example=input_example,
                        signature=mlflow.models.infer_signature(input_example, [o.detach().cpu().numpy() for o in save(input_tensor)])
                    )
                    if last[0] > best:
                        best = last[0]
                        torch.save(save.state_dict(), os.path.join(args.result_path, 'best_qat.pth'))
                        mlflow.pytorch.log_model(
                            save,
                            name="best_qat_model",
                            input_example=input_example,
                            signature=mlflow.models.infer_signature(input_example, [o.detach().cpu().numpy() for o in save(input_tensor)])
                        )
                    del save
        # Log file artifacts (optional)
        mlflow.log_artifact(os.path.join(args.result_path, 'weights_qat/step_qat.csv'))
        mlflow.log_artifact("./weight_fluc.csv")
        mlflow.log_artifact("./scale_param.txt")
    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None, device="cpu"):
    filenames = glob.glob(os.path.join(args.dataset_path, 'images/test2007/*'))
    dataset = Dataset(filenames, args.input_size, params, False)


    loader = data.DataLoader(dataset, args.batch_size, False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    
    device = torch.device(device)
    model.to(device)
    model.eval()
    # model.apply(torch.quantization.disable_observer)

    if not hasattr(model, 'nc'):
        model.nc = len(params['names'])
    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 4) % ('precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.to(device)
        samples = samples.float()  # uint8 to fp16/32
        samples = samples / 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h), device=device)
        # Inference
        outputs = model(samples)
        # print([o.shape for o in outputs])
        # # NMS
        # print(model.nc)
        outputs = non_max_suppression(outputs, 0.001, 0.7, model.nc)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.to(device)
            box = box.to(device)

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat((cls, wh2xy(box) * scale), 1)
                metric = compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)
    # Print results
    print('%10.3g' * 4 % (m_pre, m_rec, map50, mean_ap))
    # model.apply(torch.quantization.enable_observer)
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params, model):
    from thop import profile, clever_format
    shape = (1, 3, args.input_size, args.input_size)
    model.eval()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    macs, params = profile(model, inputs=(torch.zeros(shape),), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    if args.local_rank == 0:
        print("[INFO] Model profile: ")
        print(f'      MACs: {macs}')
        print(f'      Parameters: {params}')


def main():
    parser = ArgumentParser(description="Configuration to train YOLOv8 models with only-train-once pruning method.")
    parser.add_argument('--unprunable-node-id-file', type=str, default='/data/dutn1/model_compression_flow/model_compression_toolkit/experiments/YOLOv8/unprunable_node_ids_yolov8n.txt')
    parser.add_argument('--pretrained-model', type=str, default='')
    parser.add_argument('--dataset-configuration-file', type=str, default='/data/dutn1/model_compression_flow/model_compression_toolkit/experiments/YOLOv8/args_voc.yaml')
    parser.add_argument('--dataset-path', type=str, default="/data/dutn1/VSIQuantization/datasets/VOC")
    parser.add_argument('--result-path', type=str, default='/data/dutn1/VSIQuantization/check_point')
    parser.add_argument('--input-size', default=320, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.result_path, 'weights_qat')):
            os.makedirs(os.path.join(args.result_path, 'weights_qat'))

    with open(args.dataset_configuration_file, errors='ignore') as f:
        params = yaml.safe_load(f)

    # profile(args, params)
    train(args, params)
    # test(args, params)


if __name__ == "__main__":
    main()
