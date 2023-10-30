import os
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms


root = "/mnt/d/data/image/mnist/FashionMNIST/raw"

def build_dataset(root, transform=None, split="train"):
    root = os.path.join(root, split)

    if split == 'train':
        dataset = datasets.FashionMNIST(
    root="/mnt/d/data/image/mnist",
    train=True,
    download=False,
    transform=transform,
)
    else:
       dataset = datasets.FashionMNIST(
    root="/mnt/d/data/image/mnist",
    train=False,
    download=False,
    transform=transform,
)
    return dataset

train_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = build_dataset(root, train_transform, "train")
val_dataset = build_dataset(root, train_transform, "val")

bs = 64
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)

for x,y in train_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break

class Mlp(nn.Module):
    def __init__(self, in_features=718, hidden_features=512, out_features=512, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Mlp_model(nn.Module):
    def __init__(self, in_channels=28*28, hidden_features=256, depth=2, num_classes=10):
        super(Mlp_model, self).__init__()

        tem = []
        for i in range(depth):
            if i == 0:
                in_feat = in_channels
            hidden_feat = hidden_features // (i+1)
            out_feat = hidden_feat
            if i == depth-1:
                out_feat = num_classes
            tem.append(Mlp(in_feat, hidden_feat, out_feat))
            in_feat =  out_feat
        self.blocks = nn.ModuleList(tem)

    def forward(self, x):
        x = x.reshape(-1, 28*28).squeeze()
        for block in self.blocks:
            x = block(x)
        return x

model = Mlp_model().to("cuda")

for x,y in train_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    x = x.to("cuda", non_blocking=True)
    logit = model(x)
    print(logit.shape)
    break
#criterion
criterion = nn.CrossEntropyLoss()

#optimzier
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# engine
def train_one_epoch(loader, model, criterion, optimizer, len_dataset, epoch, epochs):
    
    log = {"loss": 0}
    epoch_loss, pri_freq= 0, 100
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to("cuda", non_blocking=True),  y.to("cuda", non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % pri_freq == 0:
            print(f"Epoch: [{epoch}/{epochs}] [{i}/{len(loader)}]  Loss: {loss.item():.6f}")
        
        epoch_loss += loss.item() * x.shape[0]
    epoch_loss  = epoch_loss / len_dataset
    log["loss"] = epoch_loss
    return log

def accuracy(logits, y, topk):
    bs = logits.shape[0]
    _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
    target = y.unsqueeze(dim=1).expand_as(pred)
    result = pred.eq(target)
    return result.float().sum() / bs * 100

def evaluate(loader, model, dataset_size):
    log = {"acc1":0, "acc5":0}
    epoch_acc1, epoch_acc5 = 0, 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to("cuda", non_blocking=True),  y.to("cuda", non_blocking=True)
            logits = model(x)
            top1_acc = accuracy(logits, y, 1)
            top5_acc = accuracy(logits, y, 5)
            epoch_acc1 += top1_acc 
            epoch_acc5 += top5_acc 

    epoch_acc1  = epoch_acc1 / len(loader)
    epoch_acc5  = epoch_acc5 / len(loader)

    log["acc1"] = epoch_acc1
    log["acc5"] = epoch_acc5
    return log

#main
epochs = 100
best_acc = 0
for epoch in range(epochs):
    train_log = train_one_epoch(train_loader, model, criterion, optimizer, len(train_dataset), epoch, epochs)
    val_log = evaluate(val_loader, model, len(val_dataset))
    print(f"Epoch [{epoch}/{epochs-1}] Train" + " ".join(f"{k}: {v:.6f}" for k,v in train_log.items()))
    print(f"Epoch [{epoch}/{epochs-1}] val " + " ".join(f"{k}: {v:.6f}" for k,v in val_log.items()))

    if best_acc < val_log["acc1"]:
        best_acc = val_log["acc1"]
        torch.save(model.state_dict(), f"best_{epoch}.pth")
    
print("Done")
print(f"best acc1: {best_acc:.3f}")



