import torch.nn as nn
import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class UnContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(UnContrastLoss, self).__init__()
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                Uncontrastive = d_ap / (d_an + 1e-7)
            else:
                Uncontrastive = d_ap
            loss += self.weights[i] * Uncontrastive
        return loss

def mosaic_module(input_tensor, a, b):
    B, c, H, W = input_tensor.size()

    m, n = H // a, W // b

    input_tensor = input_tensor.view(B, c, m, a, n, b)
    input_tensor = input_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
    input_tensor = input_tensor.view(B * m * n, c, a, b)

    indices = torch.randperm(B * m * n)
    input_tensor = input_tensor[indices]

    input_tensor = input_tensor.view(B, m, n, c, a, b)
    input_tensor = input_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    input_tensor = input_tensor.view(B, c, H, W)

    return input_tensor

def mosaic_module_1(input_tensor, a, b):
    B, c, H, W = input_tensor.size()

    m, n = H // a, W // b

    input_tensor = input_tensor.view(B, c, m, a, n, b)
    input_tensor = input_tensor.permute(0, 3, 5, 1, 2, 4).contiguous()
    input_tensor = input_tensor.view(B * a * b, c, m, n)

    indices = torch.randperm(B * a * b)
    input_tensor = input_tensor[indices]

    input_tensor = input_tensor.view(B, a, b, c, m, n)
    input_tensor = input_tensor.permute(0, 3, 4, 1, 5, 2).contiguous()
    input_tensor = input_tensor.view(B, c, H, W)

    return input_tensor
