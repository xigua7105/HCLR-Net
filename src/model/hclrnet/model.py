import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class HCLRNet(nn.Module):
    def __init__(self, input_nc=3, ngf=64, use_dropout=False, padding_type='reflect'):
        super(HCLRNet, self).__init__()
        ###### downsample
        self.Pad2d1 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(input_nc, ngf, kernel_size=3),
                                    nn.GELU())
        self.block1 = Attention(ngf)
        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        self.block2 = Attention(ngf)
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        self.block3 = Attention(ngf*2)
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        ###### blocks
        self.block = Block(default_conv, ngf * 4)

        ###### upsample
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU())
        self.block4 = Attention(ngf*2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU())
        self.block5 = Attention(ngf)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.GELU())
        self.block6 = Attention(ngf)
        self.Pad2d2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(ngf, 3, kernel_size=3),
                                    nn.Tanh())
        self.start_conv = default_conv(in_channels=3, out_channels=256, kernel_size=3, bias=True)
        self.Residual_block = residual_block(in_channels=256, out_channels=256, kernel_size=3)
        self.final_conv = default_conv(in_channels=256, out_channels=3, kernel_size=3, bias=True)
    def forward(self, input):

        x = self.Pad2d1(input)
        x = self.block1(x)
        x_down1 = self.down1(x)
        x_down1 = self.block2(x_down1)
        x_down2 = self.down2(x_down1)
        x_down2 = self.block3(x_down2)
        x_down3 = self.down3(x_down2)
        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)
        x_up1 = self.up1(x6)
        x_up1 = self.block4(x_up1)
        x_up1 = F.interpolate(x_up1, x_down2.size()[2:], mode='bilinear', align_corners=True)
        add1 = x_down2 + x_up1
        x_up2 = self.up2(add1)
        x_up2 = self.block5(x_up2)
        x_up2 = F.interpolate(x_up2, x_down1.size()[2:], mode='bilinear', align_corners=True)
        add2 = x_down1 + x_up2
        x_up3 = self.up3(add2)
        x_up3 = self.block6(x_up3)
        x_up3 = F.interpolate(x_up3, x.size()[2:], mode='bilinear', align_corners=True)
        add3 = x + x_up3
        result1 = self.Pad2d2(add3)

        conv = self.start_conv(input)
        Residual_block1 = self.Residual_block(conv)
        Residual_block2 = self.Residual_block(Residual_block1)
        Residual_block3 = self.Residual_block(Residual_block2)
        Residual_block4 = self.Residual_block(Residual_block3)
        Residual_block5 = self.Residual_block(Residual_block4)
        Residual_block5 = conv + Residual_block5
        result2 = self.final_conv(Residual_block5)
        result = result1 + result2
        return result



