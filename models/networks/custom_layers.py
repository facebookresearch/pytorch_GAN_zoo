# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch.nn as nn

from numpy import prod


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=bias),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)
