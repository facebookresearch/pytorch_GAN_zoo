import os
import torchvision
import torchvision.transforms as Transforms

from ..DCGAN import DCGAN
from ..dcgan_product import DCGANProduct

from .gan_trainer import GANTrainer
from .standard_configurations.dcgan_config import _C

class DCGANTrainer(GANTrainer):
    r"""
    A trainer structure for the DCGAN and DCGAN product models
    """

    def __init__(self,
                 pathdb,
                 nEpoch,
                **kwargs):
        r"""
        Args:

            pathdb (string): path to the input dataset
            nEpoch (int):    number of epoch
            **kwargs:        other arguments specific to the GANTrainer class
        """

        self.nEpoch = nEpoch
        self.tokenWindowTexture = None
        self.tokenWindowMask = None

        GANTrainer.__init__(self, pathdb, **kwargs)

        self.lossProfile.append({"G":[], "D":[], "iter":[], "scale":0})

    def getDefaultConfig(self):
        return _C

    def initModel(self):

        if self.modelConfig.productGan:
            self.model = DCGANProduct(self.modelConfig.dimLatentVectorShape,
                                      self.modelConfig.dimLatentVectorTexture,
                                      useGPU = self.useGPU,
                                      storeAVG = self.storeAVG,
                                      **vars(self.modelConfig))
        else:
            self.model = DCGAN(self.modelConfig.dimLatentVector,
                               useGPU = self.useGPU,
                               storeAVG = self.storeAVG,
                               **vars(self.modelConfig))

    def train(self):

        shift = 0
        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                        + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        for epoch in range(self.nEpoch):
            dbLoader = self.getDBLoader(0)
            self.trainOnEpoch(dbLoader, 0, shiftIter = shift)

            shift+= len(dbLoader)

    def initializeWithPretrainNetworks(self,
                                       pathD,
                                       pathGShape,
                                       pathGTexture,
                                       finetune = True):
        r"""
        Initialize a product gan by loading 3 pretrained networks

        Args:

            pathD (string): Path to the .pt file where the DCGAN discrimator is saved
            pathGShape (string): Path to .pt file where the DCGAN shape generator
                                 is saved
            pathGTexture (string): Path to .pt file where the DCGAN texture generator
                                   is saved

            finetune (bool): set to True to reinitialize the first layer of the
                             generator and the last layer of the discriminator
        """

        if not self.modelConfig.productGan:
            raise ValueError("Only product gan can be cross-initialized")

        self.model.loadG(pathGShape, pathGTexture, resetFormatLayer = finetune)
        self.model.load(pathD, loadG = False, loadD = True,
                        loadConfig = False, finetuning = True)

    def sendToVisualization(self, refVectorReal, scale, label = None):
        r"""
        Send the images generated from some reference latent vectors and a bunch
        of real examples from the dataset to the visualisation tool.
        """

        GANTrainer.sendToVisualization(self, refVectorReal, scale, label)

        if not self.modelConfig.productGan:
            return

        imgSize = max(128, refVectorReal.size()[2])

        _, shape, texture = self.model.getDetailledOutput(self.refVectorVisualization)

        self.tokenWindowTexture = self.visualisation.publishTensors(texture, (imgSize, imgSize),
                                                                    self.modelLabel + " texture",
                                                                    self.tokenWindowTexture,
                                                                    env = self.modelLabel)
        self.tokenWindowMask = self.visualisation.publishTensors(shape,
                                                            (imgSize, imgSize),
                                                            self.modelLabel + " shape",
                                                            self.tokenWindowMask,
                                                            env = self.modelLabel)
