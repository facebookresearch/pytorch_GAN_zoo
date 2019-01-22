# Progressive Product Gan
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from .progressive_gan import ProgressiveGAN
from .networks.progressive_conv_net import GNet, DNet
from .networks.product_net import ProductNetwork, \
    PRODUCT_NETWORK_ANALYSIS_MODE, \
    PRODUCT_NETWORK_FULL_MODE
from .utils.config import BaseConfig
from .utils.product_module import buildMaskSplit

from .loss_criterions.loss_texture import LossTexture


class PPGAN(ProgressiveGAN):
    r"""
    An implementation of the product network for NVIDIA's progressive GAN
    """

    def __init__(self,
                 dimTexture=1,
                 dimShape=1,
                 keySplits=None,
                 depthTexture0=0,
                 depthShape0=0,
                 maskExtraction=False,
                 mixedNoise=False,
                 textureLossModel="",
                 textureLossLayers=None,
                 weightTextureLoss=0.,
                 **kwargs):
        r"""
        Args:
            - dimTexture (int): dimension of the noise latent vector for the
                                texture generator
            - dimTexture (int): dimension of the noise latent vector for the
                                shape generator
            - keySplits (dict): if not None and ACGAN is activated, specify the
                                labels given to GShape, and the labels given to
                                GTexture.

                                We will have:

                                keySplits["GShape"] : [list of the labels for
                                                            GShape]
                                keySplits["GTexture"] : [list of the labels for
                                                            GTexture]

                                Exemple:

                                keySplits = { "GShape" : ["Size", "Type"],
                                              "GTexture" : ["Color", "Type"]}

                                If None and ACGAN is activated all labels will
                                be given to both networks
            - maskExtraction (bool): set to true to activate the mask
                                     discriminator
            - mixedNoise (bool): set to true to make both genereators use the
                                 same noise data
        """

        if 'config' not in vars(self):
            self.config = BaseConfig()

        self.config.dimRandomTexture = dimTexture
        self.config.dimRandomShape = dimShape
        self.config.keySplits = deepcopy(keySplits)

        self.config.noiseGShape = dimShape
        self.config.noiseGTexture = dimTexture

        self.config.depthTexture0 = depthTexture0
        self.config.depthShape0 = depthShape0

        self.config.productDepths = []

        self.shapeDiscrimator = None

        self.config.mixedNoise = mixedNoise

        kwargs["dimOutput"] = 3
        kwargs["dimLatentVector"] = dimTexture + dimShape

        ProgressiveGAN.__init__(self, **kwargs)

        self.lossTextureModule = None
        if textureLossModel is not "":

            if textureLossLayers is None:
                raise ValueError(
                    "Please specify the layers to extract with the texture \
                    loss model")

            self.lossTextureModule = LossTexture(
                self.device, textureLossModel, textureLossLayers)
            self.weightTextureLoss = weightTextureLoss

            if self.weightTextureLoss == 0.:
                raise ValueError(
                    "If a texture loss is applied, the weight corresponding \
                     to this loss must not be zero")

        if maskExtraction:
            print("Mask discrimator activated")
            self.shapeDiscrimator = self.getMaskDiscriminator()
            self.updateSolversDevice()

    def getNetG(self):

        if self.config.depthTexture0 == 0:
            self.config.depthTexture0 = self.config.depthScale0
        if self.config.depthShape0 == 0:
            self.config.depthShape0 = self.config.depthTexture0

        attribShift = None
        if self.config.keySplits is not None:
            attribShift = self.ACGANCriterion.getAllAttribShift()

        self.maskShape, self.maskTexture = buildMaskSplit(self.config.noiseGShape,
                                                          self.config.noiseGTexture,
                                                          self.config.categoryVectorDim,
                                                          self.config.attribKeysOrder,
                                                          attribShift,
                                                          self.config.keySplits,
                                                          mixedNoise=self.config.mixedNoise)

        self.dimLatentVectorGShape = len([x for x in self.maskShape if x > 0])
        self.dimLatentVectorGTexture = len(
            [x for x in self.maskTexture if x > 0])

        gTexture = GNet(self.dimLatentVectorGTexture,
                        self.config.depthTexture0,
                        initBiasToZero=self.config.initBiasToZero,
                        leakyReluLeak=self.config.leakyReluLeak,
                        normalization=self.config.perChannelNormalization,
                        generationActivation=self.lossCriterion.generationActivation,
                        dimOutput=3)

        gShape = GNet(self.dimLatentVectorGShape,
                      self.config.depthShape0,
                      initBiasToZero=self.config.initBiasToZero,
                      leakyReluLeak=self.config.leakyReluLeak,
                      normalization=self.config.perChannelNormalization,
                      generationActivation=self.lossCriterion.generationActivation,
                      dimOutput=1)

        # Add scales if necessary
        for depth in self.config.productDepths:
            gShape.addScale(depth[0])
            gTexture.addScale(depth[1])

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gTexture.setNewAlpha(self.config.alpha)
            gShape.setNewAlpha(self.config.alpha)

        gnet = ProductNetwork(gShape, gTexture,
                              self.maskShape, self.maskTexture)

        return gnet

    def getStateDict(self, saveTrainTmp=False):

        outdata = super(PPGAN, self).getStateDict(saveTrainTmp=saveTrainTmp)

        if self.shapeDiscrimator is not None:
            if isinstance(self.shapeDiscrimator, nn.DataParallel):
                outdata['shapeDiscrimator'] = self.shapeDiscrimator.module.state_dict()
            else:
                outdata['shapeDiscrimator'] = self.shapeDiscrimator.state_dict()

        return outdata

    def getMaskDiscriminator(self):
        r"""
        Initalize a mask discrimator to force the shape network to follow a
        given pattern.
        """
        shapeDiscrimator = DNet(self.config.depthShape0,
                                initBiasToZero=self.config.initBiasToZero,
                                leakyReluLeak=self.config.leakyReluLeak,
                                sizeDecisionLayer=1,
                                miniBatchNormalization=self.config.miniBatchStdDev,
                                dimInput=1,
                                equalizedlR=self.config.equalizedlR)

        # Do not forget the update the scale if necessary
        for depth in self.config.productDepths:
            shapeDiscrimator.addScale(depth[0])

        return shapeDiscrimator

    def updateSolversDevice(self, buildAvG=True):

        super(PPGAN, self).updateSolversDevice(buildAvG=buildAvG)

        if self.shapeDiscrimator is not None:

            if isinstance(self.shapeDiscrimator, DNet):
                self.shapeDiscrimator = nn.DataParallel(
                    self.shapeDiscrimator).to(self.device)

            self.shapeOptimizer = optim.Adam(filter(lambda p: p.requires_grad, self.shapeDiscrimator.parameters()),
                                             betas=[0., 0.99], lr=self.config.learningRate)

    def addScale(self, newDepths):
        r"""
        Add a new scale to the model. The output resolution becomes twice bigger.
        """

        depthNewScaleD, depthNewScaleG1, depthNewScaleG2 = newDepths
        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        self.netG.G1.addScale(depthNewScaleG1)
        self.netG.G2.addScale(depthNewScaleG2)
        self.netD.addScale(depthNewScaleD)

        self.config.productDepths.append((depthNewScaleG1, depthNewScaleG2))
        self.config.depthOtherScales.append(depthNewScaleD)

        if self.shapeDiscrimator is not None:

            if isinstance(self.shapeDiscrimator, nn.DataParallel):
                self.shapeDiscrimator = self.shapeDiscrimator.module
            self.shapeDiscrimator.addScale(depthNewScaleG1)

        self.updateSolversDevice()

    def getMaxScale(self):

        return len(self.config.depthOtherScales)

    def resetTmpLosses(self):

        super(PPGAN, self).resetTmpLosses()

        self.trainTmp.lossDShape = 0
        self.trainTmp.lossGShape = 0
        self.trainTmp.lossTexture = 0

    def auxiliaryLossesGeneration(self):

        if self.shapeDiscrimator is not None:

            # Discriminateur
            self.shapeOptimizer.zero_grad()
            predReal = self.shapeDiscrimator.forward(self.lastMask)
            lossDisShape = self.lossCriterion.getCriterion(predReal, True)

            inputNoiseDiscriminator, _ = self.buildNoiseData(
                self.lastMask.size(0))
            _, predFake, _ = self.netG(inputNoiseDiscriminator.to(
                self.device), mode=PRODUCT_NETWORK_FULL_MODE)
            predFake = predFake.detach()
            lossDisShape += self.lossCriterion.getCriterion(
                self.shapeDiscrimator(predFake), False)

            lossDisShape += self.getGradientPenalty(
                self.lastMask, predFake, network=self.shapeDiscrimator)
            lossDisShape += (predReal ** 2).sum() * self.config.epsilonD

            lossDisShape.backward()
            self.trainTmp.lossDShape += lossDisShape.item()

            self.shapeOptimizer.step()
            self.shapeOptimizer.zero_grad()

            inputNoise, _ = self.buildNoiseData(self.lastMask.size(0))
            _, shape, texture = self.netG(inputNoise.to(
                self.device), mode=PRODUCT_NETWORK_FULL_MODE)

            pred = self.shapeDiscrimator(shape)
            lossShape = self.lossCriterion.getCriterion(pred, True)

            self.trainTmp.lossGShape += lossShape.item()

            lossShape.backward(retain_graph=True)

        if self.lossTextureModule is not None and self.getSize() >= 128:

            lossTexture = self.weightTextureLoss * \
                self.lossTextureModule.getLoss(
                    self.real_input, texture, mask=self.lastMask)
            self.trainTmp.lossTexture += lossTexture.item()
            lossTexture.backward(retain_graph=True)

    def optimizeParameters(self, input_batch, inputLabels=None, inputMasks=None):

        # If a mask is provided, take it as a model for the shape generator
        if inputMasks is not None and self.shapeDiscrimator is not None:
            self.lastMask = inputMasks.to(self.device)

        super(PPGAN, self).optimizeParameters(input_batch, inputLabels)

    def updateAlpha(self, newAlpha):
        r"""
        Update the blending factor alpha.

        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        self.netD.setNewAlpha(newAlpha)
        self.netG.G1.setNewAlpha(newAlpha)
        self.netG.G2.setNewAlpha(newAlpha)

        self.config.alpha = newAlpha

        self.updateSolversDevice()

    def loadG(self,
              pathGShape,
              pathGTexture,
              resetFormatLayer=True):
        r"""
        Load pretrained GShape and GTexture networks.

        Args: see ProductNetwork.load
        """

        self.netG = self.getNetG()
        self.netG.load(pathGShape, pathGTexture)
        # self.netG.freezeNet("shape")

        # Don't forget to reset the machinery !
        self.updateSolversDevice()

    def loadAuxiliaryData(self, in_state):

        if 'shapeDiscrimator' in in_state:
            print("Shape discrimator detected !")
            self.shapeDiscrimator = self.getMaskDiscriminator()
            self.shapeDiscrimator.load_state_dict(in_state['shapeDiscrimator'])

    def getDetailledOutput(self, x):

        out, shape, texture = self.netG(
            x.to(self.device), mode=PRODUCT_NETWORK_ANALYSIS_MODE)
        return out.cpu(), shape.cpu(), texture.cpu()
