from .progressive_gan import ProgressiveGAN
from .networks.styleGAN import GNet
from .utils.config import BaseConfig


class StyleGAN(ProgressiveGAN):

    def __init__(self,
                 nMappings=8,
                 **kwargs):

        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.nMappings = nMappings
        ProgressiveGAN.__init__(self, **kwargs)

    def getNetG(self):

        gnet = GNet(dimInput=self.config.latentVectorDim,
                    dimMapping=self.config.depthScale0,
                    leakyReluLeak=self.config.leakyReluLeak,
                    nMappingLayers=self.config.nMappings,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=self.config.dimOutput)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet
