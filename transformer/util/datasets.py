#validate an code
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class pretrain_dataset(Dataset):
    def __init__(self, root=None, transform=None, args=None):
        self.root = root
        self.args = args
        self.transform = transform
        self.img_names_list = os.listdir(root)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.img_names_list[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        fake_label = torch.zeros([0], torch.long)

        return img, fake_label
    
    def __len__(self):
        return len(self.img_names_list)
    
if __name__== '__main__':
    train_dataset = pretrain_dataset(root="D:/data/image/COCO2017")

    for x, y in train_dataset:
        print(f"Dataset sample: iamge: {x.shape, x.dtype} label: {y.shape, y.dtype}")
        break
    