# %%
import torch.nn as nn

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, init_channels=64):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_channels = init_channels
        
        #down sampling
        self.conv1 = DoubleConv(self.in_channels, self.init_channels)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(self.init_channels , self.init_channels*2)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(self.init_channels*2, self.init_channels*4)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(self.init_channels*4, self.init_channels*8)
        self.maxpool4 = nn.MaxPool2d(2)
        
        self.bottle = DoubleConv(self.init_channels*8, self.init_channels*16)
        
        #up sampling
        self.up4 = nn.ConvTranspose2d(self.init_channels*16, self.init_channels*16 // 2, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(self.init_channels*16, self.init_channels*8)
        
        self.up3 = nn.ConvTranspose2d(self.init_channels*8, self.init_channels*8 // 2, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(self.init_channels*8, self.init_channels*4)
        
        self.up2 = nn.ConvTranspose2d(self.init_channels*4, self.init_channels*4 // 2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(self.init_channels*4, self.init_channels*2)
        
        self.up1 = nn.ConvTranspose2d(self.init_channels*2, self.init_channels*2 // 2, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(self.init_channels*2, self.init_channels)
        
        self.outconv = nn.Conv2d(self.init_channels, self.num_classes, kernel_size=1)
        
        self._init_weight()

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        bottle = self.bottle(maxpool4)
        
        up4 = self.up4(bottle)
        up4_cat = torch.cat((conv4, up4), dim=1)
        upconv4 = self.upconv4(up4_cat)
        
        up3 = self.up3(upconv4)
        up3_cat = torch.cat((conv3, up3), dim=1)
        upconv3 = self.upconv3(up3_cat)
        
        up2 = self.up2(upconv3)
        up2_cat = torch.cat((conv2, up2), dim=1)
        upconv2 = self.upconv2(up2_cat)
        
        up1 = self.up1(upconv2)
        up1_cat = torch.cat((conv1, up1), dim=1)
        upconv1 = self.upconv1(up1_cat)
        
        outconv = self.outconv(upconv1)
        
        return outconv
     
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# %%
class UNet2D_small(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, init_channels=32):
        super(UNet2D_small, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_channels = init_channels
        #down sampling
        self.conv1 = DoubleConv(self.in_channels, self.init_channels)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(self.init_channels, self.init_channels*2)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(self.init_channels*2, self.init_channels*4)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.bottle = DoubleConv(self.init_channels*4, self.init_channels*8)
        
        #up sampling
        self.up3 = nn.ConvTranspose2d(self.init_channels*8, self.init_channels*8 // 2, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(self.init_channels*8, self.init_channels*4)
        
        self.up2 = nn.ConvTranspose2d(self.init_channels*4, self.init_channels*4 // 2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(self.init_channels*4, self.init_channels*2)
        
        self.up1 = nn.ConvTranspose2d(self.init_channels*2, self.init_channels*2 // 2, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(self.init_channels*2, self.init_channels)
        
        self.outconv = nn.Conv2d(self.init_channels, self.num_classes, kernel_size=1)
        
        self._init_weight()
        
        
    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        
        bottle = self.bottle(maxpool3)
        
        up3 = self.up3(bottle)
        up3_cat = torch.cat((conv3, up3), dim=1)
        upconv3 = self.upconv3(up3_cat)
        
        up2 = self.up2(upconv3)
        up2_cat = torch.cat((conv2, up2), dim=1)
        upconv2 = self.upconv2(up2_cat)
        
        up1 = self.up1(upconv2)
        up1_cat = torch.cat((conv1, up1), dim=1)
        upconv1 = self.upconv1(up1_cat)
        
        outconv = self.outconv(upconv1)
        return outconv
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# %%


# %%
