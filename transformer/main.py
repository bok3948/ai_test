import argparse
import os
import time
import datetime
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from model_finetune import model 
import engine_finetune
from engine_finetune import train_one_epoch, evaluate

from util.datasets import pretrain_dataset
from util import misc 
from util.lr_sched import cosine_scheduler, get_params_groups

def get_args_parser():
    parser = argparse.ArgumentParser(description="ImageNet-1K classification", add_help=False)

    #hardware
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local_rank', default=0, type=int)

    #dataloader
    parser.add_argument('--data', default='D:\data\image\ILSVRC\Data\CLS-LOC')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dist_eval', default=True, type=bool)

    #model
    parser.add_argument('--model', default='vit_small', type=str)
    parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    #optimizer
    parser.add_argument('--blr', default=0.0005, type=float)
    parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--clip_grad', default=3.0, type=float)

    #run
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)

    #print
    parser.add_argument('--print_freq', default=500, type=int)

    #save
    parser.add_argument('--output_dir', default='./checkpoint_finetune', type=str)
    parser.add_argument('--log_dir', default='./log_finetune', type=str)

    return parser

def main(args):
    misc.init_distributed_mode(args)

    args.dir_name = os.path.dirname(os.path.realpath(__file__))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args).items())))

    device = torch.device(args.device)

    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    #dataset
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)

    #Sampler
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.dist_eval:
        if len(val_dataset) % num_tasks != 0:
            print("make sure len(val_dataset) % num_gpus == 0")

        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False,
    )

    eff_batch_size = args.batch_size * misc.get_world_size()
    args.num_gpus = misc.get_world_size()
    args.eff_batch_size = eff_batch_size
    print(f"effective batch size : {eff_batch_size}")

    #model
    model = model.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, args=args)

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    #Optimizer
    args.lr = args.blr * eff_batch_size / 256
    print(f"base learning rate, actual learning rate: {args.blr, args.lr}")

    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)

    #Loss
    loss_scaler = None
    criterion = None

    if args.use_fp16:
        loss_scaler = torch.cuda.amp.GradScaler()

    #scheduler
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    #Resume
    if len(args.resume) > 3:
        misc.load_model(args. model_without_ddp, optimizer, loss_scaler)

    print(f"Strat training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, 
            criterion, loss_scaler,
            lr_schedule, log_writer,
            args, 
        )
        if args.output_dir:
            misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        test_stats = evaluate(val_loader, model, device)

        print(f"Accuracy on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
def main(args):
    misc.init_distributed_mode(args)

    args.dir_name = os.path.dirname(os.path.realpath(__file__))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args).items())))

    device = torch.device(args.device)

    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    #dataset
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)

    #Sampler
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.dist_eval:
        if len(val_dataset) % num_tasks != 0:
            print("make sure len(val_dataset) % num_gpus == 0")

        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False,
    )

    eff_batch_size = args.batch_size * misc.get_world_size()
    args.num_gpus = misc.get_world_size()
    args.eff_batch_size = eff_batch_size
    print(f"effective batch size : {eff_batch_size}")

    #model
    model = model.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, args=args)

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    #Optimizer
    args.lr = args.blr * eff_batch_size / 256
    print(f"base learning rate, actual learning rate: {args.blr, args.lr}")

    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)

    #Loss
    loss_scaler = None
    criterion = None

    if args.use_fp16:
        loss_scaler = torch.cuda.amp.GradScaler()

    #scheduler
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    #Resume
    if len(args.resume) > 3:
        misc.load_model(args. model_without_ddp, optimizer, loss_scaler)

    print(f"Strat training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, 
            criterion, loss_scaler,
            lr_schedule, log_writer,
            args, 
        )
        if args.output_dir:
            misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        test_stats = evaluate(val_loader, model, device)

        print(f"Accuracy on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
def main(args):
    misc.init_distributed_mode(args)

    args.dir_name = os.path.dirname(os.path.realpath(__file__))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args).items())))

    device = torch.device(args.device)

    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    #dataset
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)

    #Sampler
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.dist_eval:
        if len(val_dataset) % num_tasks != 0:
            print("make sure len(val_dataset) % num_gpus == 0")

        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False,
    )

    eff_batch_size = args.batch_size * misc.get_world_size()
    args.num_gpus = misc.get_world_size()
    args.eff_batch_size = eff_batch_size
    print(f"effective batch size : {eff_batch_size}")

    #model
    model = model.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, args=args)

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    model = torch.compile(model)

    #Optimizer
    args.lr = args.blr * eff_batch_size / 256
    print(f"base learning rate, actual learning rate: {args.blr, args.lr}")

    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)

    #Loss
    loss_scaler = None
    criterion = None

    if args.use_fp16:
        loss_scaler = torch.cuda.amp.GradScaler()

    #scheduler
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    #Resume
    if len(args.resume) > 3:
        misc.load_model(args. model_without_ddp, optimizer, loss_scaler)

    print(f"Strat training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, 
            criterion, loss_scaler,
            lr_schedule, log_writer,
            device, epoch,
            args, 
        )
        if args.output_dir:
            misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        test_stats = evaluate(val_loader, model, device)

        print(f"Accuracy on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                      'epoch': epoch}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open('finetune_log.txt', mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)






