import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.DCGAN_nets import GNet, DNet
from .networks.product_net import ProductNetwork, PRODUCT_NETWORK_ANALYSIS_MODE
from .utils.product_module import buildMaskSplit

class DCGANProduct(BaseGAN):
    r"""
    Implementation of a product of DCGAN :
    """
    def __init__(self,
                 latentVectorDimShape = 64,
                 latentVectorDimTexture = 64,
                 dimGShape = 64,
                 dimGTexture = 64,
                 dimD = 64,
                 depth = 3,
                 keySplits = None,
                **kwargs):
        r"""
        Args:

        Specific Arguments:
            - latentVectorDimShape (int): dimension of the input latent vector
                                          of the shape generator
            - latentVectorDimTexture (int): dimension of the input latent vector
                                            of the texture generator
            - dimGShape (int): reference depth of a layer in the shape generator
            - dimGTexture (int): reference depth of a layer in the texture generator
            - dimD (int): reference depth of a layer in the discriminator
            - depth (int): number of convolution layer in the model
            - keySplits (dict): if not None and ACGAN is activated, specify the
                                labels given to GShape, and the labels given to
                                GTexture.

                                We will have:

                                keySplits["GShape"] : [list of the labels for GShape]
                                keySplits["GTexture"] : [list of the labels for GTexture]

                                Exemple:

                                keySplits = { "GShape" : ["Size", "Type"],
                                              "GTexture" : ["Color", "Type"]}

                                If None and ACGAN is activated all labels will be
                                given to both networks
            - **kwargs: arguments of the BaseGAN class

        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.noiseGShape = latentVectorDimShape
        self.config.noiseGTexture = latentVectorDimTexture

        self.config.dimGShape = dimGShape
        self.config.dimGTexture = dimGTexture
        self.config.dimD = dimD
        self.config.depth = depth
        self.config.keySplits = keySplits

        print(self.config.keySplits)

        BaseGAN.__init__(self,
                        latentVectorDimShape + latentVectorDimTexture,
                        **kwargs)

    def getNetG(self):

        self.maskShape, self.maskTexture = buildMaskSplit(self.config.noiseGShape,
                                                          self.config.noiseGTexture,
                                                          self.config.categoryVectorDim,
                                                          self.config.attribKeysOrder,
                                                          self.ACGANCriterion.getAllAttribShift(),
                                                          self.config.keySplits)


        self.dimLatentVectorGShape = len([x for x in self.maskShape if x > 0])
        self.GShape = GNet(self.dimLatentVectorGShape,
                           1,
                           self.config.dimGShape,
                           depthModel = self.config.depth,
                           generationActivation = self.lossCriterion.generationActivation)

        self.dimLatentVectorGTexture = len([x for x in self.maskTexture if x > 0])
        self.GTexture = GNet(self.dimLatentVectorGTexture,
                             self.config.dimOutput,
                             self.config.dimGTexture,
                             depthModel = self.config.depth,
                             generationActivation = self.lossCriterion.generationActivation)

        gnet = ProductNetwork(self.GShape, self.GTexture, self.maskShape, self.maskTexture)
        return gnet

    def getNetD(self):

        dnet =  DNet(self.config.dimOutput,
                     self.config.dimD,
                     self.lossCriterion.sizeDecisionLayer + self.config.categoryVectorDim,
                     depthModel = self.config.depth)
        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas = [0.1, 0.999], lr = self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas = [0.1, 0.999], lr = self.config.learningRate)

    def getSize(self):
        return 2**(self.config.depth + 3)

    def loadG(self,
              pathGShape,
              pathGTexture,
              resetFormatLayer = True):
        r"""
        Load pretrained GShape and GTexture networks.

        Args: see ProductNetwork.load
        """

        self.netG = self.getOriginalG()
        self.netG.load(pathGShape, pathGTexture, resetFormatLayer = resetFormatLayer)

        # Don't forget to reset the machinery !
        self.updateSolversDevice()

    def getDetailledOutput(self, x):

        out, shape, texture = self.netG(x.to(self.device), mode = PRODUCT_NETWORK_ANALYSIS_MODE)
        return out.cpu(), shape.cpu(), texture.cpu()
