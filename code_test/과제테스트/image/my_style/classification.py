import os

import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from model import resnet
from engine import train_one_epoch, evaluate

from util.datasets import build_dataet
from util.misc import save

#setting
output_dir = "./checkpoint"
os.makedirs(output_dir, exist_ok=True)

#transform
train_transforms = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

val_transforms = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

#dataset
#/mnt/d/data/image/ILSVRC/Data/CLS-LOC/
train_dataset = build_dataet("/mnt/d/data/image/mnist/FashionMNIST/raw", train_transforms, "train")
val_dataset = build_dataet("/mnt/d/data/image/mnist/FashionMNIST/raw", train_transforms, "val")

BATCH_SIZE = 64
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#check
for i, (x, y) in enumerate(train_loader):
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    break
print(len(train_loader))

#model
model = resnet(in_channels=1, channels=128).to(device="cuda")

#loss 
critertion = torch.nn.CrossEntropyLoss()

#optimzier
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#main
epochs = 100
best_acc, best_epoch = 0, 0
for i in range(epochs):
    train_log = train_one_epoch(train_loader, model, critertion, optimizer, len(train_dataset), i)
    print(f"Epoch [{i}/{epochs-1}] Train: " +  " ".join(f" {k}: {v:.6f}" for k, v in train_log.items()))

    val_log = evaluate(val_loader, model, critertion, len(val_dataset))
    print(f"Epoch [{i}/{epochs-1}] Val: " + " ".join(f" {k}: {v:.6f}" for k, v in val_log.items()))
    
    #save
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": i
    }
    torch.save(to_save, f"./{output_dir}/{i}.pth")


    if val_log["acc1"] > best_acc:
        best_epoch = i
        best_acc = val_log["acc1"]
        torch.save(to_save, f"./best_acc.pth")

print("Done")
print(f"Best Acc1: {best_acc} at {best_epoch}")
