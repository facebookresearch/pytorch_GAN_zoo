# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d
from ..utils.utils import num_flat_features
from.mini_batch_stddev_module import miniBatchStdDev


class GNet(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=3,
                 equalizedlR=True):
        r"""
        Build a generator for a progressive GAN model

        Args:

            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime

        """
        super(GNet, self).__init__()

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        # Initialize the scale 0
        self.initFormatLayer(dimLatent)
        self.dimOutput = dimOutput
        self.groupScale0 = nn.ModuleList()
        self.groupScale0.append(EqualizedConv2d(depthScale0, depthScale0, 3,
                                                equalized=equalizedlR,
                                                initBiasToZero=initBiasToZero,
                                                padding=1))

        self.toRGBLayers.append(EqualizedConv2d(depthScale0, self.dimOutput, 1,
                                                equalized=equalizedlR,
                                                initBiasToZero=initBiasToZero))

        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Last layer activation function
        self.generationActivation = generationActivation
        self.depthScale0 = depthScale0


    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.dimLatent = dimLatentVector
        self.formatLayer = EqualizedLinear(self.dimLatent,
                                           16 * self.scalesDepth[0],
                                           equalized=self.equalizedlR,
                                           initBiasToZero=self.initBiasToZero)

    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2

        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        depthLastScale = self.scalesDepth[-1]

        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale,
                                                    depthNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
                                                    3, padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))

        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput,
                                                1,
                                                equalized=self.equalizedlR,
                                                initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward(self, x):

        ## Normalize the input ?
        if self.normalizationLayer is not None:
            x = self.normalizationLayer(x)
        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.leakyRelu(self.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.normalizationLayer(x)

        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            x = self.leakyRelu(convLayer(x))
            if self.normalizationLayer is not None:
                x = self.normalizationLayer(x)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](x)
            y = Upscale2d(y)

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):

            x = Upscale2d(x)
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        return x


class DNet(nn.Module):

    def __init__(self,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 dimInput=3,
                 equalizedlR=True):
        r"""
        Build a discriminator for a progressive GAN model

        Args:

            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()

        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1,
                                                  equalized=equalizedlR,
                                                  initBiasToZero=initBiasToZero))

        # Minibatch standard deviation
        dimEntryScale0 = depthScale0
        if miniBatchNormalization:
            dimEntryScale0 += 1

        self.miniBatchNormalization = miniBatchNormalization
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, depthScale0,
                                                   3, padding=1,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16,
                                                   depthScale0,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

    def addScale(self, depthNewScale):

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthLastScale,
                                                    3,
                                                    padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))

        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                  depthNewScale,
                                                  1,
                                                  equalized=self.equalizedlR,
                                                  initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):

        self.decisionLayer = EqualizedLinear(self.scalesDepth[0],
                                             sizeDecisionLayer,
                                             equalized=self.equalizedlR,
                                             initBiasToZero=self.initBiasToZero)



    def forward(self, x, getFeature = False):

        # Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leakyRelu(self.fromRGBLayers[- 2](y))

        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2
        for groupLayer in reversed(self.scaleLayers):

            for layer in groupLayer:
                x = self.leakyRelu(layer(x))

            x = nn.AvgPool2d((2, 2))(x)

            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1-self.alpha) * x

            shift -= 1

        # Now the scale 0

        # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)

        x = self.leakyRelu(self.groupScaleZero[0](x))

        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))

        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x
