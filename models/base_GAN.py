from copy import deepcopy

import torch
import torch.nn as nn

from .utils.config import BaseConfig, updateConfig
from .loss_criterions import base_loss_criterions
from .loss_criterions.ac_criterion import ACGANCriterion, ChangeMyNameCriterion
from .loss_criterions.GDPP_loss import GDPPLoss
from .utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible


class BaseGAN():
    r"""Abstract class: the basic framework for GAN training.
    """

    def __init__(self,
                 dimLatentVector,
                 dimOutput=3,
                 useGPU=True,
                 kInnerD=1,
                 kInnerG=1,
                 lambdaGP=0.,
                 epsilonD=0.,
                 baseLearningRate=0.001,
                 lossMode='WGANGP',
                 attribKeysOrder=None,
                 weightConditionD=0.0,
                 weightConditionG=0.0,
                 classificationMode=None,
                 GDPP=False,
                 **kwargs):
        r"""
        Args:
            dimLatentVector (int): dimension of the latent vector in the model
            useGPU (bool): set to true if the computation should be distribued
                           in the availanle GPUs
            kInnerD (int): number of iterations for the discriminator network
                           during the training stage
            kInnerG (int): number of iterations for the generator network
                           during the training stage
            lambdaGP (float): if > 0, weight of the gradient penalty (WGANGP)
            epsilonD (float): if > 0, penalty on |D(X)|**2
            baseLearningRate (float): target learning rate.
            lossMode (string): loss used by the model. Must be one of the
                               following options
                              * 'MSE' : mean square loss. It's not advised to
                                have lambdaGP != 0 in this case
                              * 'WGANGP': cross entroipy loss.
            attribKeysOrder (dict): if not None, activate AC-GAN. In this case,
                                    both the generator and the discrimator are
                                    trained on abelled data.
        """

        if lossMode not in ['MSE', 'WGANGP', 'DCGAN']:
            raise ValueError(
                "lossMode should be one of the following : ['MSE', 'WGANGP', \
                'DCGAN']")

        if classificationMode not in [None, 'ACGAN', 'ChangeMyName']:
            raise ValueError(
                "lossMode should be one of the following : ['MSE', 'WGANGP', \
                'DCGAN']")

        if 'config' not in vars(self):
            self.config = BaseConfig()

        if 'trainTmp' not in vars(self):
            self.trainTmp = BaseConfig()

        self.useGPU = useGPU and torch.cuda.is_available()
        if self.useGPU:
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1

        # Latent vector dimension
        self.config.noiseVectorDim = dimLatentVector

        # Output image dimension
        self.config.dimOutput = dimOutput

        # Actual learning rate
        self.config.learningRate = baseLearningRate

        # AC-GAN ?
        self.config.attribKeysOrder = deepcopy(attribKeysOrder)
        self.config.categoryVectorDim = 0
        self.config.weightConditionG = weightConditionG
        self.config.weightConditionD = weightConditionD
        self.ClassificationCriterion = None
        self.initializeClassificationCriterion(classificationMode)

        # GDPP
        self.config.GDPP = GDPP

        self.config.latentVectorDim = self.config.noiseVectorDim \
            + self.config.categoryVectorDim

        # Loss criterion
        self.config.lossCriterion = lossMode
        self.lossCriterion = getattr(
            base_loss_criterions, lossMode)(self.device)

        # WGAN-GP
        self.config.lambdaGP = lambdaGP

        # Weight on D's output
        self.config.epsilonD = epsilonD

        # Initialize the generator and the discriminator
        self.netD = self.getNetD()
        self.netG = self.getNetG()

        # Move the networks to the gpu
        self.updateSolversDevice()

        # Inner iterations
        self.config.kInnerD = kInnerD
        self.config.kInnerG = kInnerG


    def test(self, input, getAvG=False, toCPU=True):
        r"""
        Generate some data given the input latent vector.

        Args:
            input (torch.tensor): input latent vector
        """
        input = input.to(self.device)
        if getAvG:
            if toCPU:
                return self.avgG(input).cpu()
            else:
                return self.avgG(input)
        elif toCPU:
            return self.netG(input).detach().cpu()
        else:
            return self.netG(input).detach()

    def buildAvG(self):
        r"""
        Create and upload a moving average generator.
        """
        self.avgG = deepcopy(self.getOriginalG())
        for param in self.avgG.parameters():
            param.requires_grad = False

        if self.useGPU:
            self.avgG = nn.DataParallel(self.avgG)
            self.avgG.to(self.device)

    def optimizeParameters(self, input_batch, inputLabels=None):
        r"""
        Update the discrimator D using the given "real" inputs.

        Args:
            input (torch.tensor): input batch of real data

        """

        allLosses = {}

        # Retrieve the input data
        self.real_input, self.realLabels = input_batch.to(self.device), None

        if self.config.attribKeysOrder is not None:
            self.realLabels = inputLabels.to(self.device)

        n_samples = self.real_input.size()[0]

        # Update the discriminator
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD(self.real_input, False)

        # Classification criterion
        predRealD, allLosses["lossD_classif"] = \
            self.classificationPenalty(predRealD,
                                       self.realLabels,
                                       self.config.weightConditionD,
                                       backward=True)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples,
                                                         inputLabels = self.realLabels)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)

        predFakeD, _ = self.classificationPenalty(predFakeD,
                                                  targetRandCat,
                                                  0,
                                                  backward=False)

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD += lossDFake

        # #3 WGANGP loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_Grad"] = self.getGradientPenalty(self.real_input,
                                                              predFakeG,
                                                              backward=True,
                                                              labels=targetRandCat)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (predRealD[:, 0] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        lossD.backward()
        finiteCheck(self.netD.module.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputNoise, targetCatNoise = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputNoise)

        # #2 Status evaluation
        predFakeD, phiGFake = self.netD(predFakeG, True)

        # #2 Classification criterion
        predFakeD, allLosses["lossG_classif"] = \
            self.classificationPenalty(predFakeD,
                                       targetCatNoise,
                                       self.config.weightConditionG,
                                       backward=True)

        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses["lossG_fake"] = lossGFake.item()
        lossGFake.backward()

        if self.config.GDPP:
            _, phiDReal = self.netD.forward(self.real_input, True)
            allLosses["lossG_GDPP"] = GDPPLoss(phiDReal, phiGFake,
                                               backward=True)

        finiteCheck(self.netG.module.parameters())
        self.optimizerG.step()

        lossG = 0
        for key, val in allLosses.items():

            if key.find("lossG") == 0:
                lossG += val

        allLosses["lossG"] = lossG

        # Update the moving average if relevant
        for p, avg_p in zip(self.netG.module.parameters(),
                            self.avgG.module.parameters()):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return allLosses

    def initializeClassificationCriterion(self, classificationMode):
        r"""
        """

        if self.config.weightConditionD != 0 and \
                not self.config.attribKeysOrder:
            raise AttributeError("If the weight on the conditional term isn't "
                                 "null, then a attribute dictionnery should be"
                                 " defined")

        if self.config.weightConditionG != 0 and \
                not self.config.attribKeysOrder:
            raise AttributeError("If the weight on the conditional term isn't \
                                 null, then a attribute dictionnary should be \
                                 defined")

        if self.config.attribKeysOrder is not None:
            if classificationMode == 'ACGAN':
                self.ClassificationCriterion = \
                        ACGANCriterion(self.config.attribKeysOrder)
            else:
                self.ClassificationCriterion = \
                        ChangeMyNameCriterion(self.config.attribKeysOrder)

            self.config.categoryVectorDim = \
                self.ClassificationCriterion.getInputDim()

    def updateSolversDevice(self, buildAvG=True):
        r"""
        Move the current networks and solvers to the GPU.
        This function must be called each time netG or netD is modified
        """
        if self.buildAvG():
            self.buildAvG()

        if not isinstance(self.netD, nn.DataParallel) and self.useGPU:
            self.netD = nn.DataParallel(self.netD)
        if not isinstance(self.netG, nn.DataParallel) and self.useGPU:
            self.netG = nn.DataParallel(self.netG)

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizerD = self.getOptimizerD()
        self.optimizerG = self.getOptimizerG()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()

    def buildNoiseData(self, n_samples, inputLabels = None):
        r"""
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """

        inputLatent = torch.randn(
            n_samples, self.config.noiseVectorDim).to(self.device)

        if self.config.attribKeysOrder:

            if inputLabels is not None:
                latentRandCat = self.ClassificationCriterion.buildLatentCriterion(inputLabels)
                targetRandCat = inputLabels
            else:
                targetRandCat, latentRandCat = \
                    self.ClassificationCriterion.buildRandomCriterionTensor(n_samples)

            targetRandCat = targetRandCat.to(self.device)
            latentRandCat = latentRandCat.to(self.device)

            inputLatent = torch.cat((inputLatent, latentRandCat), dim=1)

            return inputLatent, targetRandCat

        return inputLatent, None

    def buildNoiseDataWithConstraints(self, n, labels):

        constrainPart = \
            self.ClassificationCriterion.generateConstraintsFromVector(n, labels)
        inputLatent = torch.randn((n, self.config.noiseVectorDim, 1, 1))

        return torch.cat((inputLatent, constrainPart), dim=1)

    def getOriginalG(self):
        r"""
        Retrieve the original G network. Use this function
        when you want to modify G after the initialization
        """
        if isinstance(self.netG, nn.DataParallel):
            return self.netG.module
        return self.netG

    def getOriginalD(self):
        r"""
        Retrieve the original D network. Use this function
        when you want to modify D after the initialization
        """
        if isinstance(self.netD, nn.DataParallel):
            return self.netD.module
        return self.netD

    def getNetG(self):
        r"""
        The generator should be defined here.
        """
        pass

    def getNetD(self):
        r"""
        The discrimator should be defined here.
        """
        pass

    def getOptimizerD(self):
        r"""
        Optimizer of the discriminator.
        """
        pass

    def getOptimizerG(self):
        r"""
        Optimizer of the generator.
        """
        pass

    def getStateDict(self, saveTrainTmp=False):
        r"""
        Get the model's parameters
        """
        # Get the generator's state
        stateG = self.getOriginalG().state_dict()

        # Get the discrimator's state
        stateD = self.getOriginalD().state_dict()

        out_state = {'config': self.config,
                     'netG': stateG,
                     'netD': stateD}

        # Average GAN
        out_state['avgG'] = self.avgG.module.state_dict()

        if saveTrainTmp:
            out_state['tmp'] = self.trainTmp

        return out_state

    def save(self, path, saveTrainTmp=False):
        r"""
        Save the model at the given location.

        All parameters included in the self.config class will be saved as well.
        Args:
            - path (string): file where the model should be saved
            - saveTrainTmp (bool): set to True if you want to conserve
                                    the training parameters
        """
        torch.save(self.getStateDict(saveTrainTmp=saveTrainTmp), path)

    def updateConfig(self, config):
        r"""
        Update the object config with new inputs.

        Args:

            config (dict or BaseConfig) : fields of configuration to be updated

            Typically if config = {"learningRate": 0.1} only the learning rate
            will be changed.
        """
        updateConfig(self.config, config)
        self.updateSolversDevice()

    def load(self,
             path="",
             in_state=None,
             loadG=True,
             loadD=True,
             loadConfig=True,
             finetuning=False):
        r"""
        Load a model saved with the @method save() function

        Args:
            - path (string): file where the model is stored
        """

        in_state = torch.load(path)
        self.load_state_dict(in_state,
                             loadG = loadG,
                             loadD = loadD,
                             loadConfig=True,
                             finetuning=False)

    def load_state_dict(self,
                        in_state,
                        loadG=True,
                        loadD=True,
                        loadConfig=True,
                        finetuning=False):
        r"""
        Load a model saved with the @method save() function

        Args:
            - in_state (dict): state dict containing the model
        """

        # Step one : load the configuration
        if loadConfig:
            updateConfig(self.config, in_state['config'])
            self.lossCriterion = getattr(
                base_loss_criterions, self.config.lossCriterion)(self.device)
            self.initializeClassificationCriterion()

        # Re-initialize G and D with the loaded configuration
        buildAvG = True

        if loadG:
            self.netG = self.getNetG()
            if finetuning:
                loadPartOfStateDict(
                    self.netG, in_state['netG'], ["formatLayer"])
                self.getOriginalG().initFormatLayer(self.config.latentVectorDim)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netG, in_state['netG'])
                if 'avgG' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    # Replace me by a standard loadStatedict for open-sourcing
                    loadStateDictCompatible(self.avgG.module, in_state['avgG'])
                    buildAvG = False

        if loadD:

            self.netD = self.getNetD()
            if finetuning:
                loadPartOfStateDict(
                    self.netD, in_state['netD'], ["decisionLayer"])
                self.getOriginalD().initDecisionLayer(
                    self.lossCriterion.sizeDecisionLayer
                    + self.config.categoryVectorDim)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netD, in_state['netD'])

        elif 'tmp' in in_state.keys():
            self.trainTmp = in_state['tmp']

        # Don't forget to reset the machinery !
        self.updateSolversDevice(buildAvG)

    def classificationPenalty(self, outputD, target, weight, backward=True):
        r"""
        """

        if self.ClassificationCriterion is not None:
            outputD, loss = self.ClassificationCriterion.getCriterion(outputD,
                                                                      target)
            if loss is not None:
                loss *= weight
                if backward:
                    loss.backward(retain_graph=True)
                return outputD, loss.item()
        return outputD, 0

    def getGradientPenalty(self, input, fake, backward=True, labels=None):
        r"""
        Build the gradient penalty as described in
        "Improved Training of Wasserstein GANs"

        Args:

            - input (Tensor): batch of real data
            - fake (Tensor): batch of generated data. Must have the same size
              as the input
        """

        batchSize = input.size(0)
        alpha = torch.rand(batchSize, 1)
        alpha = alpha.expand(batchSize, int(input.nelement() /
                                            batchSize)).contiguous().view(
                                                input.size())
        alpha = alpha.to(self.device)
        interpolates = alpha * input + ((1 - alpha) * fake)

        interpolates = torch.autograd.Variable(
            interpolates, requires_grad=True)

        decisionInterpolate = self.netD(interpolates, False)

        if labels is None:
            decisionInterpolate = decisionInterpolate[:, 0]
        else:
            decisionInterpolate, _ = \
                self.classificationPenalty(decisionInterpolate, labels,
                                           0, backward=False)

        decisionInterpolate = decisionInterpolate.sum()

        gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            decisionInterpolate.size()).to(
                                                            self.device),
                                        create_graph=True, retain_graph=True,
                                        only_inputs=True)

        gradients = gradients[0].view(batchSize, -1)
        gradients = (gradients * gradients).sum(dim=1).sqrt()
        gradient_penalty = (((gradients - 1.0)**2)).sum() * \
            self.config.lambdaGP

        if backward:
            gradient_penalty.backward(retain_graph=True)

        return gradient_penalty.item()
