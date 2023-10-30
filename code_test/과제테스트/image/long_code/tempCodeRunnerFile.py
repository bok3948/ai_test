import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms

torch.manual_seed(1)
root = "/mnt/d/data/image/프로그래머스문제"

class MinMaxScaling(object):
    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        return Image.fromarray((image * 255).astype('uint8'))

class dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = []
        label_map = {
        "0" : -2,
        'dog' : 0,
        'elephant' : 1,
        'giraffe' : 2,
        'guitar' : 3,
        'horse' : 4,
        'house' : 5,
        'person' : 6,}
        for path, _, filenames in os.walk(self.root):
            label_name = os.path.basename(path)
            label = label_map.get(label_name, -1)
            if label != -1:
                for filename in filenames:
                    if filename.lower().endswith("jpg"):
                        img_path = os.path.join(path, filename)
                        self.imgs.append((img_path, label))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, y = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        return img, y


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Totensor(),
])

train_dataset = dataset(os.path.join(root,"train/train"), transform=train_transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = dataset(os.path.join(root,"test/test"), transform=train_transform)
for x,y in train_dataset:
    print(x.shape, y)
    break

for x,y in test_dataset:
    print(x.shape)
    break

print(len(train_dataset), len(val_dataset),len(test_dataset))

bs = 128
train_loader = DataLoader(train_dataset, bs, shuffle=True)
val_loader = DataLoader(val_dataset, 10, shuffle=False)
test_loader = DataLoader(test_dataset, 1, shuffle=False)

for x, y in val_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break

#model
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_features)
        ) if stride > 1 else None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual) 
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, depth=4, in_channels=3, hidden_features=64, num_classes=7):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_features, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(hidden_features, hidden_features ,1)
        self.layer2 = self._make_layer(hidden_features, hidden_features*2 ,2)
        self.layer3 = self._make_layer(hidden_features*2, hidden_features*4 ,2)
        self.layer4 = self._make_layer(hidden_features*4, hidden_features*8, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_features*8, num_classes)

    def _make_layer(self, in_feat, out_feat, stride):
        tem = []
        tem.append(BasicBlock(in_feat, out_feat, stride))
        tem.append(BasicBlock(out_feat, out_feat, 1))
        return nn.Sequential(*tem)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

model = ResNet().to("cuda")
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
torch.autograd.set_detect_anomaly(True)
best_acc, best_epoch = 0, 0
epochs = 5
for epoch in range(1, epochs+1):
    train_one_epoch(train_loader, model, criterion, optimizer, epoch, epochs)
    val_log = evaluate(val_loader, model)
    print(f"val [{epoch}/{epochs}] " + " ".join(
        f" {k}: {v:.3f}" for k, v in val_log.items()
    ))

    if best_acc < val_log["acc1"]:
        best_acc = val_log["acc1"]
        best_epoch = epoch
        torch.save(model.state_dict(), "/mnt/d/code/code_test/과제테스트/image/long_code/best.pth")
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
    
inference(model, "/mnt/d/code/code_test/과제테스트/image/long_code/best.pth", test_loader)