import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets
from torchvision import models
import torchvision.transforms as transforms

torch.manual_seed(1)
root = "/mnt/d/data/image/프로그래머스문제"

train_dataset = datasets.ImageFolder(os.path.join(root,"train/train"), transform=transforms.ToTensor())
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = datasets.ImageFolder(os.path.join(root,"test/test"), transform=transforms.ToTensor())
for x,y in train_dataset:
    print(x.shape, y)
    break

for x,y in test_dataset:
    print(x.shape)
    break

print(len(train_dataset), len(val_dataset),len(test_dataset))

bs = 128
train_loader = DataLoader(train_dataset, bs, shuffle=True)
val_loader = DataLoader(val_dataset, 1, shuffle=False)
sampler_test = torch.utils.data.SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, 1, sampler=sampler_test, shuffle=False)

for x, y in val_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break


model = models.resnet18(pretrained=True)
num_classes = 7
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to("cuda")
print(model)

for x, y in train_loader:
    x = x.to("cuda")
    logits = model(x)
    print(logits.shape)
    break

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), 0.001)

def train_one_epoch(loader, model, criterion, optimizer, epoch, epochs):
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to("cuda", non_blocking=True),  y.to("cuda", non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"train [{epoch}/ {epochs}] [{i}/{len(loader)}] loss: {loss.item():.6f}")
    return

def evaluate(loader, model):
    log = {"acc1": 0}
    epoch_acc1 = 0
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            x, y = x.to("cuda", non_blocking=True),  y.to("cuda", non_blocking=True)
            logits = model(x)
            acc1 = (logits.argmax(dim=1) == y).float().sum() / x.shape[0] * 100
            epoch_acc1 += acc1
    epoch_acc1 = epoch_acc1 / len(loader)
    log["acc1"] = epoch_acc1
    return log

best_acc, best_epoch = 0, 0
epochs = 50
for epoch in range(1, epochs+1):
    train_one_epoch(train_loader, model, criterion, optimizer, epoch, epochs)
    val_log = evaluate(val_loader, model)
    print(f"val [{epoch}/{epochs}] " + " ".join(
        f" {k}: {v:.3f}" for k, v in val_log.items()
    ))

    if best_acc < val_log["acc1"]:
        best_acc = val_log["acc1"]
        best_epoch = epoch
        torch.save(model.state_dict(), "/mnt/d/code/code_test/과제테스트/image/short_code/best.pth")
print("Done")
print(f"Best acc: {best_acc:.3f} at {best_epoch}")


def inference(model, checkpoint_root, loader):

    checkpoint = torch.load(checkpoint_root, map_location='cpu') 
    msg = model.load_state_dict(checkpoint)
    print(msg)
    ans = []
    for x, _ in loader:
        x = x.to("cuda", non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        ans.append(int(pred))
        df = pd.DataFrame()
        df["answer value"] = ans
        df.to_csv("/mnt/d/data/image/프로그래머스문제/test_answer.csv")
    
inference(model, "/mnt/d/code/code_test/과제테스트/image/short_code/best.pth", test_loader)