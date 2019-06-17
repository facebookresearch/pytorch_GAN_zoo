# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GNet(nn.Module):
    def __init__(self,
                 dimLatentVector,
                 dimOutput,
                 dimModelG,
                 depthModel=3,
                 generationActivation=nn.Tanh()):
        super(GNet, self).__init__()

        self.depthModel = depthModel
        self.refDim = dimModelG

        self.initFormatLayer(dimLatentVector)

        currDepth = int(dimModelG * (2**depthModel))

        sequence = OrderedDict([])
        # input is Z, going into a convolution
        sequence["batchNorm0"] = nn.BatchNorm2d(currDepth)
        sequence["relu0"] = nn.ReLU(True)

        for i in range(depthModel):

            nextDepth = int(currDepth / 2)

            # state size. (currDepth) x 2**(i+1) x 2**(i+1)
            sequence["convTranspose" + str(i+1)] = nn.ConvTranspose2d(
                currDepth, nextDepth, 4, 2, 1, bias=False)
            sequence["batchNorm" + str(i+1)] = nn.BatchNorm2d(nextDepth)
            sequence["relu" + str(i+1)] = nn.ReLU(True)

            currDepth = nextDepth

        sequence["outlayer"] = nn.ConvTranspose2d(
            dimModelG, dimOutput, 4, 2, 1, bias=False)

        self.outputAcctivation = generationActivation

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

    def initFormatLayer(self, dimLatentVector):

        currDepth = int(self.refDim * (2**self.depthModel))
        self.formatLayer = nn.ConvTranspose2d(
            dimLatentVector, currDepth, 4, 1, 0, bias=False)

    def forward(self, input):

        x = input.view(-1, input.size(1), 1, 1)
        x = self.formatLayer(x)
        x = self.main(x)

        if self.outputAcctivation is None:
            return x

        return self.outputAcctivation(x)


class DNet(nn.Module):
    def __init__(self,
                 dimInput,
                 dimModelD,
                 sizeDecisionLayer,
                 depthModel=3):
        super(DNet, self).__init__()

        currDepth = dimModelD
        sequence = OrderedDict([])

        # input is (nc) x 2**(depthModel + 3) x 2**(depthModel + 3)
        sequence["convTranspose" +
                 str(depthModel)] = nn.Conv2d(dimInput, currDepth,
                                              4, 2, 1, bias=False)
        sequence["relu" + str(depthModel)] = nn.LeakyReLU(0.2, inplace=True)

        for i in range(depthModel):

            index = depthModel - i - 1
            nextDepth = currDepth * 2

            # state size.
            # (currDepth) x 2**(depthModel + 2 -i) x 2**(depthModel + 2 -i)
            sequence["convTranspose" +
                     str(index)] = nn.Conv2d(currDepth, nextDepth,
                                             4, 2, 1, bias=False)
            sequence["batchNorm" + str(index)] = nn.BatchNorm2d(nextDepth)
            sequence["relu" + str(index)] = nn.LeakyReLU(0.2, inplace=True)

            currDepth = nextDepth

        self.dimFeatureMap = currDepth

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

        self.initDecisionLayer(sizeDecisionLayer)

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = nn.Conv2d(
            self.dimFeatureMap, sizeDecisionLayer, 4, 1, 0, bias=False)
        self.decisionLayer.apply(weights_init)
        self.sizeDecisionLayer = sizeDecisionLayer

    def forward(self, input, getFeature = False):
        x = self.main(input)

        if getFeature:

            return self.decisionLayer(x).view(-1, self.sizeDecisionLayer), \
                   x.view(-1, self.dimFeatureMap * 16)

        x = self.decisionLayer(x)
        return x.view(-1, self.sizeDecisionLayer)
