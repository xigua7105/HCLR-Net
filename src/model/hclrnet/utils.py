import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def catcat(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 2)
class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x1, x2):
        return catcat(x1, x2)

class Block(nn.Module):
    def __init__(self, conv, dim):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, 3, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, 1, bias=True)
        self.act2 = nn.GELU()
        self.conv3 = conv(dim, dim, 3, bias=True)
        self.attention = Attention(dim)
    def forward(self, x):
        res1 = self.act1(self.conv1(x))
        res2 = self.act2(self.conv2(x))
        res = res1 + res2
        res = x + res
        res = self.attention(res)
        res = self.conv3(res)
        res = x + res
        return res

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.max = nn.AdaptiveMaxPool2d((1, None))
        self.conv1x1 = default_conv(dim, dim//2, kernel_size=1, bias=True)
        self.conv3x3 = default_conv(dim//2, dim, kernel_size=3, bias=True)
        self.con3x3 = default_conv(dim, dim, kernel_size=3, bias=True)
        self.GELU = nn.GELU()
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_h = self.avg(x)
        x_w = self.max(x)
        x_h = torch.squeeze(x_h, 3)
        x_w = torch.squeeze(x_w, 2)
        x_h1 = x_h.unsqueeze(3)
        x_w1 = x_w.unsqueeze(2)
        x_h_w = catcat(x_h, x_w)
        x_h_w = x_h_w.unsqueeze(3)
        x_h_w = self.conv1x1(x_h_w)
        x_h_w = self.GELU(x_h_w)
        x_h_w = torch.squeeze(x_h_w, 3)
        x1, x2 = torch.split(x_h_w, [height, width], 2)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(2)
        x1 = self.conv3x3(x1)
        x2 = self.conv3x3(x2)
        mix1 = self.mix1(x_h1, x1)
        mix2 = self.mix2(x_w1, x2)
        x1 = self.con3x3(mix1)
        x2 = self.con3x3(mix2)
        matrix = torch.matmul(x1, x2)
        matrix = torch.sigmoid(matrix)
        final = torch.mul(x, matrix)
        final = x + final
        return final

class se_block(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(se_block, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, inplanes//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.se(x)
        return input*x

class residual_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super(residual_block, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True),
            se_block(out_channels, reduction=16)
        )

    def forward(self, x):
        input = x
        x = self.residual(x)
        return input + x
