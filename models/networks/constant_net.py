# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class ConstantNet(nn.Module):
    r"A network that does nothing"

    def __init__(self,
                 shapeOut=None):

        super(ConstantNet, self).__init__()
        self.shapeOut = shapeOut

    def forward(self, x):

        if self.shapeOut is not None:
            x = x.view(x.size[0], self.shapeOut[0],
                       self.shapeOut[1], self.shapeOut[2])

        return x


class MeanStd(nn.Module):
    def __init__(self):
        super(MeanStd, self).__init__()

    def forward(self,x):

        # Size : N C W H
        x = x.view(x.size(0), x.size(1), -1)
        mean_x = torch.mean(x, dim=2)
        var_x = torch.mean(x**2, dim=2) - mean_x * mean_x
        return torch.cat([mean_x, var_x], dim=1)


class FeatureTransform(nn.Module):
    r"""
    Concatenation of a resize tranform and a normalization
    """

    def __init__(self,
                 mean=None,
                 std=None,
                 size=224):

        super(FeatureTransform, self).__init__()
        self.size = size

        if mean is None:
            mean = [0., 0., 0.]

        if std is None:
            std = [1., 1., 1.]

        self.register_buffer('mean', torch.tensor(
            mean, dtype=torch.float).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            std, dtype=torch.float).view(1, 3, 1, 1))

        if size is None:
            self.upsamplingModule = None
        else:
            self.upsamplingModule = torch.nn.Upsample(
                (size, size), mode='bilinear')

    def forward(self, x):

        if self.upsamplingModule is not None:
            x = self.upsamplingModule(x)

        x = x - self.mean
        x = x / self.std

        return x
