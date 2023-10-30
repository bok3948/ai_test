import torch
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(resblock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_feat)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x



class resnet(nn.Module):
    def __init__(self, in_channels=3 , channels=10, depth=3, classes=1000):
        super(resnet, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU())
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*(2**(depth)), classes, 1, 1, 0)
                                  )
        tem = []

        for i in range(depth):
            if i == 0:
                dim = channels
            tem.append(resblock(dim, dim*2))
            dim *= 2

        self.blocks = nn.ModuleList(tem)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x



        