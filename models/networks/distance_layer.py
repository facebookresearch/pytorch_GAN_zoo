import torch
import torch.nn as nn

class DistanceLayer(nn.Module):

    def __init__(self, size):

        super(ConstrainedLayer, self).__init__()
        self.weights = torch.randn(size)
        self.weights.requires_grad = True

    def forward(self, x):

        return (self.weights -x).norm()
