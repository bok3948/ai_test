import os
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
test_dataset = datasets.ImageFolder(os.path.join(root,"test/test"), transform=transforms.ToTensor())

test_loader = DataLoader(test_dataset, 1, shuffle=False)

for x, y in test_loader:
    print(x.shape, y)

def inference(checkpoint_root, loader):
    model = resnet(in_channels=3, depth=4, hidden_features=64, num_classes=7)
    checkpoint = torch.load(checkpoint_root, map_location='cpu')
    
    msg = model.load_state_dict(checkpoint)
    print(msg)
    ans = []
    for x, _ in loader:
        logits = model(x)
        pred = logits.argmax(dim=1) 
        print(pred)
        ans.append(int(pred))
        df = pd.DataFrame()
        df["answer value"] = ans
        df.to_csv("/mnt/d/data/image/프로그래머스문제/test_answer.csv")

#inference("/home/taeho/code/code_test/classfication/best.pth", test_loader)