import math
import sys
from typing import Iterable

import torch

from timm.utils import accuracy

import util.misc as misc

def train_one_epoch(data_loader: Iterable, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loss_scaler,
                lr_scheduler=None, 
                device=None, epoch=None, log_writer=None,
                args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="   ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='value:.6f'))
    header = 'Epoch [{}]'.format(epoch)
    
    for it, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it]
            
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if loss_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = misc.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            loss_scaler.scale(loss).backward()
            if args.clip_grad:
                loss_scaler.unscale_(optimizer)
                param_norms = misc.clip_gradients(model, args.clip_grad)

            loss_scaler.step(optimizer)
            loss_scaler.update()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            log_writer.add_scalar('loss', loss_value_reduce, epoch)
            log_writer.add_scalar('lr', lr, epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.alobal_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'Test:'

    for inputs, labels in metric_logger.log_every(data_loader, 100, header):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)

        loss_value = loss.item()
        
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1:.3f} Acc@5 {metric_logger.acc5:.3f} loss {metric_logger.loss:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







