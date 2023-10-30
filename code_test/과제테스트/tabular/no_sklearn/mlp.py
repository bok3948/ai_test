import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler

root = "/mnt/d/data/tabular"
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"
sample_root = "/mnt/d/data/tabular/tit_submission.csv"

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
sample = pd.read_csv(sample_root)

def check(df):
    print("-"*50)
    print()
    df.info()
for de in [train, test, sample]:
    check(de)

Y = train["Survived"]
train.drop(columns=["Survived"], axis=1, inplace=True)

test_PassengerId = test["PassengerId"]

#EDA
#print(train["Name"][:5])
def name_processor(x):
    ans = x.split('.')[0].split()[1]
    return ans
tem_train = pd.DataFrame()  
tem_train["title"] = train["Name"].apply(name_processor)
print(Y.groupby(tem_train["title"]).mean())
print(tem_train["title"].value_counts())
title_list = ["Mr", "Miss", "Mrs", "Master"]

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

def feat_cls(df):
    a, b, c, d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == 'object') and (len(df[col].unique()) > 10):
            a.append(col)
        elif (df[col].dtype == 'object') and (len(df[col].unique()) <= 10):
            b.append(col)
        elif not(df[col].dtype == 'object') and (len(df[col].unique()) > 10):
            c.append(col)
        elif not(df[col].dtype == 'object') and (len(df[col].unique()) <= 10):
            d.append(col)
    print(f"too_many_cat: {a}\n cat: {b}\n numer: {c}\n int_cat: {d}")
    return a,b,c,d

a,b,c,d = feat_cls(df)

df["title"] = df["Name"].apply(name_processor)

def remove_key(df, name_list):
    a = set(df)
    for dfdf in name_list:
        a.remove(dfdf)
    return a
rare = remove_key(df["title"], ["Mr", "Miss", "Mrs", "Master"])

df["title"] = df["title"].replace(rare, "rare")
df["name_len"] = df["Name"].apply(lambda x: len(x))
df["len_ticket"] = df["Ticket"].apply(lambda x: len(x))
print(df["title"].value_counts())

df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Pclass"] = df["Pclass"].astype(str)

df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass", "title"])
df = df.astype(float)
a,b,c,d= feat_cls(df)

x_train = df[:len_train].to_numpy()
x_test = df[len_train:].copy().to_numpy()
print(x_train.shape)
Y = Y.to_numpy()

class dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.label is not None:
            y = self.label[idx]
        else:
            y = 0
        data, y = torch.tensor(data, dtype=torch.float32),torch.tensor(y, dtype=torch.int64)

        return data, y
    
train_dataset = dataset(x_train, Y)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = dataset(x_test, None)


bs = 64
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
sampler_test = torch.utils.data.SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, 1, sampler=sampler_test, shuffle=False)

for x,y in train_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break

class Mlp(nn.Module):
    def __init__(self, in_features=718, hidden_features=512, out_features=512, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU(inplace=True)
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
    def __init__(self, in_channels=19, hidden_features=64, depth=2, num_classes=2):
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
        #x = x.reshape(-1, 28*28).squeeze()
        for block in self.blocks:
            x = block(x)
        return x

model = Mlp_model().to("cuda")
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
epochs = 1000
for epoch in range(1, epochs+1):
    train_one_epoch(train_loader, model, criterion, optimizer, epoch, epochs)
    val_log = evaluate(val_loader, model)
    print(f"val [{epoch}/{epochs}] " + " ".join(
        f" {k}: {v:.3f}" for k, v in val_log.items()
    ))

    if best_acc < val_log["acc1"]:
        best_acc = val_log["acc1"]
        best_epoch = epoch
        torch.save(model.state_dict(), "/mnt/d/code/code_test/과제테스트/tabular/no_sklearn/best.pth")
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
    sub = pd.DataFrame()
    sub["PassengerId"] = test_PassengerId
    sub["Survived"] = ans
    sub.to_csv(os.path.join(root, "submission.csv"), index=False)
inference(model,  "/mnt/d/code/code_test/과제테스트/tabular/no_sklearn/best.pth", test_loader)