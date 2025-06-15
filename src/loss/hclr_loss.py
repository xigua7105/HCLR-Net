import torch
from torch import nn
from .ucr import UnContrastLoss, mosaic_module
from .l1 import L1loss
from .perc import PerceptualLoss


class HCLRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = L1loss()
        self.perc = PerceptualLoss()
        self.ucr = UnContrastLoss()

        self.weights = {
            'l1': 1.0,
            'perc': 0.2,
            'ucr': 0.2,
        }

    def forward(self, outputs, targets, unpairs):
        outputs = outputs.to(targets.dtype)
        unpairs = mosaic_module(unpairs, 16, 16)

        losses = {
            'l1': self.l1(outputs, targets),
            'perc': self.perc(outputs, targets),
            'ucr': self.ucr(outputs, targets, unpairs)
        }

        final_loss =sum(self.weights[k] * v for k, v in losses.items())
        losses['loss'] = final_loss
        return losses
