# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from ..DCGAN import DCGAN
from .gan_trainer import GANTrainer
from .standard_configurations.dcgan_config import _C


class DCGANTrainer(GANTrainer):
    r"""
    A trainer structure for the DCGAN and DCGAN product models
    """

    _defaultConfig = _C

    def getDefaultConfig(self):
        return DCGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 **kwargs):
        r"""
        Args:

            pathdb (string): path to the input dataset
            **kwargs:        other arguments specific to the GANTrainer class
        """

        GANTrainer.__init__(self, pathdb, **kwargs)

        self.lossProfile.append({"iter": [], "scale": 0})

    def initModel(self):
        self.model = DCGAN(useGPU=self.useGPU,
                           **vars(self.modelConfig))

    def train(self):

        shift = 0
        if self.startIter >0:
            shift+= self.startIter

        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        maxShift = int(self.modelConfig.nEpoch * len(self.getDBLoader(0)))

        for epoch in range(self.modelConfig.nEpoch):
            dbLoader = self.getDBLoader(0)
            self.trainOnEpoch(dbLoader, 0, shiftIter=shift)

            shift += len(dbLoader)

            if shift > maxShift:
                break

        label = self.modelLabel + ("_s%d_i%d" %
                                   (0, shift))
        self.saveCheckpoint(self.checkPointDir,
                            label, 0, shift)

    def initializeWithPretrainNetworks(self,
                                       pathD,
                                       pathGShape,
                                       pathGTexture,
                                       finetune=True):
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

        self.model.loadG(pathGShape, pathGTexture, resetFormatLayer=finetune)
        self.model.load(pathD, loadG=False, loadD=True,
                        loadConfig=False, finetuning=True)
