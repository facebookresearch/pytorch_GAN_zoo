from .progressive_gan import ProgressiveGAN
from .networks.styleGAN import GNet
from .utils.config import BaseConfig


class StyleGAN(ProgressiveGAN):

    def __init__(self,
                 nMappings=8,
                 phiTruncation=0.5,
                 gamma_avg=0.99,
                 **kwargs):

        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.nMappings = nMappings
        self.config.phiTruncation = phiTruncation
        self.config.gamma_avg = gamma_avg

        if self.config.phiTruncation >= 1:
            print("Disabling the truncation trick")
        ProgressiveGAN.__init__(self, **kwargs)

    def getNetG(self):

        gnet = GNet(dimInput=self.config.latentVectorDim,
                    dimMapping=self.config.depthScale0,
                    leakyReluLeak=self.config.leakyReluLeak,
                    nMappingLayers=self.config.nMappings,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=self.config.dimOutput,
                    phiTruncation=self.config.phiTruncation,
                    gamma_avg=self.config.gamma_avg)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet
