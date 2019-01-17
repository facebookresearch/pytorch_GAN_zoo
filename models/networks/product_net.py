import torch
import torch.nn as nn

from ..utils.utils import loadPartOfStateDict

import sys

PRODUCT_NETWORK_DEFAULT_MODE = 0
PRODUCT_NETWORK_FULL_MODE = 1
PRODUCT_NETWORK_ANALYSIS_MODE = 2

class ProductNetwork(nn.Module):
    r"""
    A generator performing the product of two subnetworks:
    - a shape generator
    - an alpha mask
    """

    def __init__(self,
                 G1,
                 G2,
                 maskG1,
                 maskG2):
        r"""
        Args:

            G1 (nn.Module) : shape generator. Should output an image with 1 channel.
            G2 (nn.Module) : texture generator. Should output an image of the same
                             size as G1's.
            maskG1 (list) : 8bit mask to apply to the input latent vector to get
                            G1's input. We must have:
                                sum([x for x in maskG1 if x ==1]) == dimInputLatentVectorG1
            maskG2 (list) : 8bit mask to apply to the input latent vector to get
                            G2's input. We must have:
                                sum([x for x in maskG2 if x ==1]) == dimInputLatentVectorG2
        """

        super(ProductNetwork, self).__init__()

        self.G1 = G1
        self.G2 = G2

        self.register_buffer('maskG1', torch.tensor(maskG1, dtype = torch.uint8).view(1, -1))
        self.register_buffer('maskG2', torch.tensor(maskG2, dtype = torch.uint8).view(1, -1))

    def forward(self, x, mode = PRODUCT_NETWORK_DEFAULT_MODE):

        x1 = x[self.maskG1.expand(x.size(0), -1)]
        x2 = x[self.maskG2.expand(x.size(0), -1)]

        x1 = x1.view(x.size()[0], -1, 1, 1)
        x2 = x2.view(x.size()[0], -1, 1, 1)

        y1 = self.G1.forward(x1)
        y2 = self.G2.forward(x2)

        y1 = y1 *0.5 + 0.5

        y1 = torch.clamp(y1, min=0, max =1)

        out  = y1.expand(-1, y2.size()[1], y1.size()[2], y1.size()[3]) * y2 \
             + 1.0 - y1.expand(-1, y2.size()[1], y1.size()[2], y1.size()[3])

        if mode == PRODUCT_NETWORK_DEFAULT_MODE:
            return out
        elif mode == PRODUCT_NETWORK_FULL_MODE:
            return out, 2 * (y1 - 0.5), y2
        else:
            y1 = y1.expand(-1, y2.size()[1], y1.size()[2], y1.size()[3])
            return out.detach(), 2 * (y1.detach() - 0.5), y2.detach()

    def getDimLatentShape(self):
        return len([x for x in self.maskG1[0,:] if x > 0])

    def getDimLatentTexture(self):
        return len([x for x in self.maskG2[0,:] if x > 0])

    def freezeNet(self, index, value = False):

        if index == "shape":
            for param in self.G1.parameters():
                param.requires_grad = value
        elif index == "texture":
            for param in self.G2.parameters():
                param.requires_grad = value
        else:
            raise ValueError("index must be in {'shape', 'texture'}")

    def getOutputSize(self):
        return self.G1.getOutputSize()

    def load(self,
             pathGShape,
             pathGTexture,
             resetShape = True,
             resetTexture = True):
        r"""
        Load pretrained GShape and GTexture networks.

        Args:

            pathGShape (string): path to the '.pt' file where the shape GAN is
                                 saved. This must be a DCGAN instance.

            pathGShape (string): path to the '.pt' file where the texture GAN is
                                 saved. This must be a DCGAN instance.

            resetFormatLayer (bool): if True, then the first layer of the shape
                                     and the texture generator will be reinitialized.

                                     If you load a model with a different input
                                     latent dimension than your current's, then
                                     this parameter must be set to True.
        """

        forbiddenLayers = ["formatLayer"]
        in_stateShape = torch.load(pathGShape)

        if resetShape:
            loadPartOfStateDict(self.G1, in_stateShape['netG'], forbiddenLayers)
            self.G1.initFormatLayer(self.getDimLatentShape())
        else:
            self.G1.load_state_dict(in_stateShape['netG'])

        in_stateTexture = torch.load(pathGTexture)
        if resetTexture:
            loadPartOfStateDict(self.G2, in_stateTexture['netG'], forbiddenLayers)
            self.G2.initFormatLayer(self.getDimLatentTexture())
        else:
            self.G2.load_state_dict(in_stateTexture['netG'])
