import torch
import json
from .progressive_gan_trainer import ProgressiveGANTrainer
from ..pp_gan import PPGAN
from ..utils.utils import getMinOccurence
from .standard_configurations.ppgan_config import _C

class PPGANTrainer(ProgressiveGANTrainer):

    _defaultConfig = _C
    def getDefaultConfig(self):
        return PPGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 maskProfile = None,
                 **kwargs):

        ProgressiveGANTrainer.__init__(self, pathdb, **kwargs)

        self.dimLatentVector = self.model.config.latentVectorDim
        self.tokenWindowMask = None
        self.tokenWindowTexture = None

        self.maskProfile = {}
        if maskProfile is not None:
            self.maskProfile = {int(scale): path for scale, path in maskProfile.items()}

    def initModel(self):

        self.model = PPGAN(self.modelConfig.latentTexture,
                           self.modelConfig.latentShape,
                           useGPU = self.useGPU,
                           depthScale0 = self.modelConfig.depthScales[0][0],
                           depthTexture0 = self.modelConfig.depthScales[0][2],
                           depthShape0 = self.modelConfig.depthScales[0][1],
                           **vars(self.modelConfig))

    def updateDatasetForScale(self, scale):

        ProgressiveGANTrainer.updateDatasetForScale(self, scale)
        self.pathDBMask = getMinOccurence(self.maskProfile, scale, self.pathDBMask)

    def initializeWithPretrainNetworks(self,
                                       pathD,
                                       pathGShape,
                                       pathGTexture,
                                       pathConfig,
                                       finetune = True):
        r"""
        Initialize a product gan by loading 3 pretrained networks

        Args:

            pathD (string): Path to the .pt file where the DCGAN discrimator is saved
            pathGShape (string): Path to .pt file where the DCGAN shape generator
                                 is saved
            pathGTexture (string): Path to .pt file where the DCGAN texture generator
                                   is saved
            pathConfig (string): path to the configuration file on the new model

            finetune (bool): set to True to reinitialize the first layer of the
                             generator and the last layer of the discriminator
        """

        # Read the training configuration
        #trainConfig = json.load(open(pathConfig, 'rb'))
        #self.readTrainConfig(trainConfig)
        #self.initModel()

        for depths in self.modelConfig.depthScales[1:]:
            self.model.addScale(depths)

        self.startScale = len(self.modelConfig.depthScales) -1

        self.model.load(pathD, loadG = False, loadD = True,
                        loadConfig = False, finetuning = True)
        self.model.loadG(pathGShape, pathGTexture, resetFormatLayer = finetune)

    def sendToVisualization(self, refVectorReal, scale, **kwargs):
        r"""
        Send the images generated from some reference latent vectors and a bunch
        of real examples from the dataset to the visualisation tool.
        """

        ProgressiveGANTrainer.sendToVisualization(self, refVectorReal, scale, **kwargs)
        imgSize = max(128, refVectorReal.size()[2])

        _, texture, shape = self.model.getDetailledOutput(self.refVectorVisualization)

        self.tokenWindowTexture = self.visualisation.publishTensors(texture, (imgSize, imgSize),
                                                            self.modelLabel + " texture",
                                                            self.tokenWindowTexture,
                                                            env = self.modelLabel)
        self.tokenWindowMask = self.visualisation.publishTensors(shape,
                                                            (imgSize, imgSize),
                                                            self.modelLabel + " shape",
                                                            self.tokenWindowMask,
                                                            env = self.modelLabel)
