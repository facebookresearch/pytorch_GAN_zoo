from ..styleGAN import StyleGAN
from .progressive_gan_trainer import ProgressiveGANTrainer

from .standard_configurations.stylegan_config import _C

class StyleGANTrainer(ProgressiveGANTrainer):

    _defaultConfig = _C

    def getDefaultConfig(self):
        return StyleGANTrainer._defaultConfig

    def __init__(self, pathdb, **kwargs):
        ProgressiveGANTrainer.__init__(self, pathdb, **kwargs)

    def initModel(self):
        config = {key: value for key, value in vars(self.modelConfig).items()}
        config["depthScale0"] = self.modelConfig.depthScales[0]
        self.model = StyleGAN(useGPU=self.useGPU, **config)
        if self.startScale ==0:
            self.startScale = 1
            self.model.addScale(self.modelConfig.depthScales[1])
