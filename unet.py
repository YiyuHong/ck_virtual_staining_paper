

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
ref.
https://github.com/milesial/Pytorch-UNet/

'''

class DoubleConv(nn.Module):
    ''' (Conv -> BN -> Relu) -> (Conv -> BN -> Relu) '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )

    def forward(self, x):
        return self.double_conv(x)

#class DoubleConv(nn.Module):
#    ''' (Conv -> BN -> Relu) -> (Conv -> BN -> Relu) '''
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.double_conv = nn.Sequential(
#                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                nn.InstanceNorm2d(out_channels),
#                nn.ReLU(inplace = True),
#                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                nn.InstanceNorm2d(out_channels),
#                nn.ReLU(inplace=True),
#                )
#    def forward(self, x):
#        return self.double_conv(x)
#    
    

class Encode(nn.Module):
    '''Encode : Downscaling with maxpooling then double conv'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block_conv(x)

class Decode(nn.Module):
    '''Decode : Upscaling with linear or Conv method'''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.decode = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.decode = nn.ConvTranspose2d(in_channels//2,
                                             in_channels//2,
                                             kernel_size=2,
                                             stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.decode(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        #corping
#        x2 = x2[:, diffY // 2 : diffY - diffY // 2, diffX // 2 : diffX - diffX // 2 ]
        #padding
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

#
class OutConv(nn.Module):
    '''Output conv layer with 1*1 conv'''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return torch.tanh(self.conv(x))

class Unet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_chnnels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.encode1 = Encode(64, 128)
        self.encode2 = Encode(128, 256)
        self.encode3 = Encode(256, 512)
        self.encode4 = Encode(512, 512)
        self.decode1 = Decode(1024, 256, bilinear)
        self.decode2 = Decode(512, 128, bilinear)
        self.decode3 = Decode(256, 64, bilinear)
        self.decode4 = Decode(128, 64, bilinear)
        self.out_conv = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)
        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)
        logits = self.out_conv(x)
        return logits

class UnetAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UnetAttention, self).__init__()
        self.n_chnnels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.encode1 = Encode(64, 128)
        self.encode2 = Encode(128, 256)
        self.encode3 = Encode(256, 512)
        self.encode4 = Encode(512, 512)
        self.decode1 = Decode(1024, 256, bilinear)
        self.decode2 = Decode(512, 128, bilinear)
        self.decode3 = Decode(256, 64, bilinear)
        self.decode4 = Decode(128, 64, bilinear)
        self.Attention = nn.Conv2d(64, 1, kernel_size=1)
        self.out_conv = OutConv(64 + 1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)
        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x_h = self.decode4(x, x1)
        x_a = self.Attention(x_h)
        x_o = torch.cat([x_h, x_a], dim=1)
        logits = self.out_conv(x_o)
        out = torch.cat([logits, x_a], dim=1)
        return out

### short run
#import numpy as np
#import matplotlib.pyplot as plt
#np.random.normal()
#np.ones((3,256,256))
#img = np.random.normal(0, 1, 256*256*3).reshape(3,256,256)
#plt.imshow(img.transpose(1,2,0))
#
##img_torch = torch.tensor(img.reshape(1,3,512,512)).double()
#img_torch = torch.from_numpy(img.reshape(1,3,256,256)).double()
#u_net = Unet(3,1,True).double()
#output_img=u_net(img_torch)
#
#output_img.shape
#output_img = output_img.detach().numpy()
#output_img_squ = output_img.squeeze()
#plt.imshow(output_img_squ)

