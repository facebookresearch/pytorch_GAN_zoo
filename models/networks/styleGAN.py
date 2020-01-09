import torch
import torch.nn as nn

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d


class AdaIN(nn.Module):

    def __init__(self, dimIn, dimOut, epsilon=1e-8):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = EqualizedLinear(dimIn, 2*dimOut, equalized=True,
                                              initBiasToZero=True)
        self.dimOut = dimOut

    def forward(self, x, y):

        # x: N x C x W x H
        batchSize, nChannel, width, height = x.size()
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        varx = torch.clamp((tmpX*tmpX).mean(dim=2).view(batchSize, nChannel, 1, 1) - mux*mux, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - mux) * varx

        # Adapt style
        styleY = self.styleModulator(y)
        yA = styleY[:, : self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)

        return yA * x + yB


class NoiseMultiplier(nn.Module):

    def __init__(self):
        super(NoiseMultiplier, self).__init__()
        self.module = nn.Conv2d(1, 1, 1, bias=False)
        self.module.weight.data.fill_(0)

    def forward(self, x):

        return self.module(x)


class MappingLayer(nn.Module):

    def __init__(self, dimIn, dimLatent, nLayers, leakyReluLeak=0.2):
        super(MappingLayer, self).__init__()
        self.FC = nn.ModuleList()

        inDim = dimIn
        for i in range(nLayers):
            self.FC.append(EqualizedLinear(inDim, dimLatent, lrMul=0.01, equalized=True, initBiasToZero=True))
            inDim = dimLatent

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

    def forward(self, x):
        for layer in self.FC:
            x = self.activation(layer(x))

        return x

class GNet(nn.Module):

    def __init__(self,
                 dimInput=512,
                 dimMapping=512,
                 dimOutput=3,
                 nMappingLayers=8,
                 leakyReluLeak=0.2,
                 generationActivation=None,
                 phiTruncation=0.5,
                 gamma_avg=0.99):

        super(GNet, self).__init__()
        self.dimMapping = dimMapping
        self.mapping = MappingLayer(dimInput, dimMapping, nMappingLayers)
        self.baseScale0 = nn.Parameter(torch.ones(1, dimMapping, 4, 4), requires_grad=True)

        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        self.noiseModulators = nn.ModuleList()
        self.depthScales = [dimMapping]
        self.noramlizationLayer = NormalizationLayer()

        self.adain00 = AdaIN(dimMapping, dimMapping)
        self.noiseMod00 = NoiseMultiplier()
        self.adain01 = AdaIN(dimMapping, dimMapping)
        self.noiseMod01 = NoiseMultiplier()
        self.conv0 = EqualizedConv2d(dimMapping, dimMapping, 3, equalized=True,
                                     initBiasToZero=True, padding=1)

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)
        self.alpha = 0
        self.generationActivation = generationActivation
        self.dimOutput = dimOutput
        self.phiTruncation = phiTruncation

        self.register_buffer('mean_w', torch.randn(1, dimMapping))
        self.gamma_avg = gamma_avg

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

    def addScale(self, dimNewScale):

        lastDim = self.depthScales[-1]
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(lastDim,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))

        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.scaleLayers[-1].append(EqualizedConv2d(dimNewScale,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.toRGBLayers.append(EqualizedConv2d(dimNewScale,
                                                self.dimOutput,
                                                1,
                                                equalized=True,
                                                initBiasToZero=True))

        self.noiseModulators.append(nn.ModuleList())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.depthScales.append(dimNewScale)

    def forward(self, x):

        batchSize = x.size(0)
        mapping = self.mapping(self.noramlizationLayer(x))
        if self.training:
            self.mean_w = self.gamma_avg * self.mean_w + (1-self.gamma_avg) * mapping.mean(dim=0, keepdim=True)

        if self.phiTruncation < 1:
            mapping = self.mean_w + self.phiTruncation * (mapping - self.mean_w)

        feature = self.baseScale0.expand(batchSize, -1, 4, 4)
        feature = feature + self.noiseMod00(torch.randn((batchSize, 1, 4, 4), device=x.device))

        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)
        feature = feature + self.noiseMod01(torch.randn((batchSize, 1, 4, 4), device=x.device))
        feature = self.activation(feature)
        feature = self.adain01(feature, mapping)

        for nLayer, group in enumerate(self.scaleLayers):

            noiseMod = self.noiseModulators[nLayer]
            feature = Upscale2d(feature)
            feature = group[0](feature) + noiseMod[0](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[1](feature, mapping)
            feature = group[2](feature) + noiseMod[1](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[3](feature, mapping)

            if self.alpha > 0 and nLayer == len(self.scaleLayers) -2:
                y = self.toRGBLayers[-2](feature)
                y = Upscale2d(y)

        feature = self.toRGBLayers[-1](feature)
        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            feature = self.alpha * y + (1.0-self.alpha) * feature

        if self.generationActivation is not None:
            feature = self.generationActivation(feature)

        return feature

    def getOutputSize(self):

        side =  2**(2 + len(self.toRGBLayers))
        return (side, side)
