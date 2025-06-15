import torch


class L1loss(torch.nn.Module):
    """L1 loss."""
    def __init__(self):
        super(L1loss, self).__init__()

    def forward(self, X, Y):
        loss = torch.mean(torch.abs(X-Y))
        return loss
