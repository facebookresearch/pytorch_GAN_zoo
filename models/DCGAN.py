# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.optim as optim

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.DCGAN_nets import GNet, DNet


class DCGAN(BaseGAN):
    r"""
    Implementation of DCGAN
    """

    def __init__(self,
                 dimLatentVector=64,
                 dimG=64,
                 dimD=64,
                 depth=3,
                 **kwargs):
        r"""
        Args:

        Specific Arguments:
            - latentVectorDim (int): dimension of the input latent vector
            - dimG (int): reference depth of a layer in the generator
            - dimD (int): reference depth of a layer in the discriminator
            - depth (int): number of convolution layer in the model
            - **kwargs: arguments of the BaseGAN class

        """
        if 'config' not in vars(self):
            self.config = BaseConfig()

        self.config.dimG = dimG
        self.config.dimD = dimD
        self.config.depth = depth

        BaseGAN.__init__(self, dimLatentVector, **kwargs)

    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.dimOutput,
                    self.config.dimG,
                    depthModel=self.config.depth,
                    generationActivation=self.lossCriterion.generationActivation)
        return gnet

    def getNetD(self):

        dnet = DNet(self.config.dimOutput,
                    self.config.dimD,
                    self.lossCriterion.sizeDecisionLayer
                    + self.config.categoryVectorDim,
                    depthModel=self.config.depth)
        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0.5, 0.999], lr=self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0.5, 0.999], lr=self.config.learningRate)

    def getSize(self):
        size = 2**(self.config.depth + 3)
        return (size, size)
