import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_


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
test_loader = DataLoader(test_dataset, 1, shuffle=False)

for x, y in val_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break

#model
class resblock(nn.Module):
    def __init__(self, in_features, out_features, stride):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.residual = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride, 1),
            nn.BatchNorm2d(out_features)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.residual(residual)
        return self.act(x)

class resnet(nn.Module):
    def __init__(self, in_channels, depth, hidden_features, num_classes):
        super(resnet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, 7, 2, 3, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        tem = []
        for i in range(depth):
            tem.append(resblock(hidden_features, hidden_features*2, 2))
            tem.append(resblock(hidden_features*2, hidden_features*2, 1))
            hidden_features = hidden_features*2
        
        self.blocks = nn.ModuleList(tem)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_features, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x
from torchvision import models
import torch

model = models.resnet18(pretrained=True)
num_classes = 7
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
#model = resnet(in_channels=3, depth=4, hidden_features=256, num_classes=7).to("cuda")
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
epochs = 5
for epoch in range(1, epochs+1):
    train_one_epoch(train_loader, model, criterion, optimizer, epoch, epochs)
    val_log = evaluate(val_loader, model)
    print(f"val : [{epoch}/{epochs}] " + " ".join(
        f" {k}: {v:.3f}" for k, v in val_log.items()
    ))

    if best_acc < val_log["acc1"]:
        best_acc = val_log["acc1"]
        best_epoch = epoch
        torch.save(model.state_dict(), "/home/taeho/code/code_test/classfication/best.pth")
print("Done")
print(f"Best acc: {best_acc:.3f} at {best_epoch}")


def inference(model, checkpoint_root, loader):

    checkpoint = torch.load(checkpoint_root, map_location='cpu') 
    model = model.to("cpu")
    msg = model.load_state_dict(checkpoint)
    print(msg)
    ans = []
    for x, y in loader:
        #x, y = x.to("cuda", non_blocking=True),  y.to("cuda", non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)

        #acc1 = (logits.argmax(dim=1) == y).float().sum() / x.shape[0] * 100

        ans.append(int(pred))
        df = pd.DataFrame()
        df["answer value"] = ans
        df.to_csv("/mnt/d/data/image/프로그래머스문제/test_answer.csv")
    

inference(model, "/home/taeho/code/code_test/classfication/best.pth", val_loader)