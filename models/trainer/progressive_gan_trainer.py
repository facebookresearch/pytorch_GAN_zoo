# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from .standard_configurations.pgan_config import _C
from ..progressive_gan import ProgressiveGAN
from .gan_trainer import GANTrainer
from ..utils.utils import getMinOccurence
import torch.nn.functional as F


class ProgressiveGANTrainer(GANTrainer):
    r"""
    A class managing a progressive GAN training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """
    _defaultConfig = _C

    def getDefaultConfig(self):
        return ProgressiveGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 miniBatchScheduler=None,
                 datasetProfile=None,
                 configScheduler=None,
                 **kwargs):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
                               dataset
            - useGPU (bool): set to True if you want to use the available GPUs
                             for the training procedure
            - visualisation (module): if not None, a visualisation module to
                                      follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
                                        model'sloss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
                              (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the checkpoints
                                      should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary. See std_p_gan_config.py
                                   for all the possible options
            - numWorkers (int): number of GOU to use. Will be set to one if not
                                useGPU
            - stopOnShitStorm (bool): should we stop the training if a diverging
                                     behavior is detected ?
        """

        self.configScheduler = {}
        if configScheduler is not None:
            self.configScheduler = {
                int(key): value for key, value in configScheduler.items()}

        self.miniBatchScheduler = {}
        if miniBatchScheduler is not None:
            self.miniBatchScheduler = {
                int(x): value for x, value in miniBatchScheduler.items()}

        self.datasetProfile = {}
        if datasetProfile is not None:
            self.datasetProfile = {
                int(x): value for x, value in datasetProfile.items()}

        GANTrainer.__init__(self, pathdb, **kwargs)

    def initModel(self):
        r"""
        Initialize the GAN model.
        """

        config = {key: value for key, value in vars(self.modelConfig).items()}
        config["depthScale0"] = self.modelConfig.depthScales[0]
        self.model = ProgressiveGAN(useGPU=self.useGPU, **config)

    def readTrainConfig(self, config):
        r"""
        Load a permanent configuration describing a models. The variables
        described in this file are constant through the training.
        """

        GANTrainer.readTrainConfig(self, config)

        if self.modelConfig.alphaJumpMode not in ["custom", "linear"]:
            raise ValueError(
                "alphaJumpMode should be one of the followings: \
                'custom', 'linear'")

        if self.modelConfig.alphaJumpMode == "linear":

            self.modelConfig.alphaNJumps[0] = 0
            self.modelConfig.iterAlphaJump = []
            self.modelConfig.alphaJumpVals = []

            self.updateAlphaJumps(
                self.modelConfig.alphaNJumps, self.modelConfig.alphaSizeJumps)

        self.scaleSanityCheck()

    def scaleSanityCheck(self):

        # Sanity check
        n_scales = min(len(self.modelConfig.depthScales),
                       len(self.modelConfig.maxIterAtScale),
                       len(self.modelConfig.iterAlphaJump),
                       len(self.modelConfig.alphaJumpVals))

        self.modelConfig.depthScales = self.modelConfig.depthScales[:n_scales]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale[:n_scales]
        self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump[:n_scales]
        self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals[:n_scales]

        self.modelConfig.size_scales = [4]
        for scale in range(1, n_scales):
            self.modelConfig.size_scales.append(
                self.modelConfig.size_scales[-1] * 2)

        self.modelConfig.n_scales = n_scales

    def updateAlphaJumps(self, nJumpScale, sizeJumpScale):
        r"""
        Given the number of iterations between two updates of alpha at each
        scale and the number of updates per scale, build the effective values of
        self.maxIterAtScale and self.alphaJumpVals.

        Args:

            - nJumpScale (list of int): for each scale, the number of times
                                        alpha should be updated
            - sizeJumpScale (list of int): for each scale, the number of
                                           iterations between two updates
        """

        n_scales = min(len(nJumpScale), len(sizeJumpScale))

        for scale in range(n_scales):

            self.modelConfig.iterAlphaJump.append([])
            self.modelConfig.alphaJumpVals.append([])

            if nJumpScale[scale] == 0:
                self.modelConfig.iterAlphaJump[-1].append(0)
                self.modelConfig.alphaJumpVals[-1].append(0.0)
                continue

            diffJump = 1.0 / float(nJumpScale[scale])
            currVal = 1.0
            currIter = 0

            while currVal > 0:

                self.modelConfig.iterAlphaJump[-1].append(currIter)
                self.modelConfig.alphaJumpVals[-1].append(currVal)

                currIter += sizeJumpScale[scale]
                currVal -= diffJump

            self.modelConfig.iterAlphaJump[-1].append(currIter)
            self.modelConfig.alphaJumpVals[-1].append(0.0)

    def inScaleUpdate(self, iter, scale, input_real):

        if self.indexJumpAlpha < len(self.modelConfig.iterAlphaJump[scale]):
            if iter == self.modelConfig.iterAlphaJump[scale][self.indexJumpAlpha]:
                alpha = self.modelConfig.alphaJumpVals[scale][self.indexJumpAlpha]
                self.model.updateAlpha(alpha)
                self.indexJumpAlpha += 1

        if self.model.config.alpha > 0:
            low_res_real = F.avg_pool2d(input_real, (2, 2))
            low_res_real = F.upsample(
                low_res_real, scale_factor=2, mode='nearest')

            alpha = self.model.config.alpha
            input_real = alpha * low_res_real + (1-alpha) * input_real

        return input_real

    def updateDatasetForScale(self, scale):

        self.modelConfig.miniBatchSize = getMinOccurence(
            self.miniBatchScheduler, scale, self.modelConfig.miniBatchSize)
        self.path_db = getMinOccurence(
            self.datasetProfile, scale, self.path_db)

        # Scale scheduler
        if self.configScheduler is not None:
            if scale in self.configScheduler:
                print("Scale %d, updating the training configuration" % scale)
                print(self.configScheduler[scale])
                self.model.updateConfig(self.configScheduler[scale])

    def train(self):
        r"""
        Launch the training. This one will stop if a divergent behavior is
        detected.

        Returns:

            - True if the training completed
            - False if the training was interrupted due to a divergent behavior
        """

        n_scales = len(self.modelConfig.depthScales)

        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        for scale in range(self.startScale, n_scales):

            self.updateDatasetForScale(scale)

            while scale >= len(self.lossProfile):
                self.lossProfile.append(
                    {"scale": scale, "iter": []})

            dbLoader = self.getDBLoader(scale)
            sizeDB = len(dbLoader)

            shiftIter = 0
            if self.startIter > 0:
                shiftIter = self.startIter
                self.startIter = 0

            shiftAlpha = 0
            while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                    self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                shiftAlpha += 1

            while shiftIter < self.modelConfig.maxIterAtScale[scale]:

                self.indexJumpAlpha = shiftAlpha
                status = self.trainOnEpoch(dbLoader, scale,
                                           shiftIter=shiftIter,
                                           maxIter=self.modelConfig.maxIterAtScale[scale])

                if not status:
                    return False

                shiftIter += sizeDB
                while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                        self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                    shiftAlpha += 1

            # Save a checkpoint
            if self.checkPointDir is not None:
                realIter = min(
                    shiftIter, self.modelConfig.maxIterAtScale[scale])
                label = self.modelLabel + ("_s%d_i%d" %
                                           (scale, realIter))
                self.saveCheckpoint(self.checkPointDir,
                                    label, scale, realIter)
            if scale == n_scales - 1:
                break

            self.model.addScale(self.modelConfig.depthScales[scale + 1])

        self.startScale = n_scales
        self.startIter = self.modelConfig.maxIterAtScale[-1]
        return True

    def addNewScales(self, configNewScales):

        if configNewScales["alphaJumpMode"] not in ["custom", "linear"]:
            raise ValueError("alphaJumpMode should be one of the followings: \
                            'custom', 'linear'")

        if configNewScales["alphaJumpMode"] == 'custom':
            self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump + \
                configNewScales["iterAlphaJump"]
            self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals + \
                configNewScales["alphaJumpVals"]

        else:
            self.updateAlphaJumps(configNewScales["alphaNJumps"],
                                  configNewScales["alphaSizeJumps"])

        self.modelConfig.depthScales = self.modelConfig.depthScales + \
            configNewScales["depthScales"]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale + \
            configNewScales["maxIterAtScale"]

        self.scaleSanityCheck()
