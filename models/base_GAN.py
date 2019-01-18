from copy import deepcopy

import os
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.config import BaseConfig, updateConfig
from .loss_criterions import base_loss_criterions
from .loss_criterions.ac_criterion import ACGanCriterion
from .utils.utils import loadPartOfStateDict, finiteCheck


def getNArgs(x):

    sizeX = x.size()
    out = 1
    for s in sizeX:
        out *= s

    return out


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
                 **kwargs):
        r"""
        Args:
            dimLatentVector (int): dimension of the latent vector in the model
            useGPU (bool): set to true if the computation should be distribued
                           in the availanle GPUs
            kInnerD (int): number of iterations for the discriminator network
                           during the training stage
            kInnerG (int): number of iterations for the generator network during
                           the training stage
            lambdaGP (float): if > 0, weight of the gradient penalty (WGANGP)
            epsilonD (float): if > 0, penalty on |D(X)|**2
            baseLearningRate (float): target learning rate.
            lossMode (string): loss used by the model. Must be one of the
                               following options
                              * 'MSE' : mean square loss. It's not advised to
                                have lambdaGP != 0 in this case
                              * 'WGANGP': cross entroipy loss.
            attribKeysOrder (dict): if not None, activate AC-GAN. In this case, both the generator and
                                   the discrimator are trained on abelled data.
        """

        if lossMode not in ['MSE', 'WGANGP', 'DCGAN']:
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

        # AC-GAN ?
        self.config.attribKeysOrder = deepcopy(attribKeysOrder)
        self.config.categoryVectorDim = 0
        self.config.weightConditionG = weightConditionG
        self.config.weightConditionD = weightConditionD
        self.initializeACCriterion()

        self.config.latentVectorDim = self.config.noiseVectorDim \
            + self.config.categoryVectorDim

        # Loss criterion
        self.config.lossCriterion = lossMode
        self.lossCriterion = getattr(
            base_loss_criterions, lossMode)(self.device)

        # Initialize the generator and the discriminator
        self.netD = self.getNetD()
        self.netG = self.getNetG()

        # Actual learning rate
        self.config.learningRate = baseLearningRate

        # Move the networks to the gpu
        self.updateSolversDevice()

        # Inner iterations
        self.config.kInnerD = kInnerD
        self.config.kInnerG = kInnerG

        # Set the inner k iteration to zero
        self.trainTmp.currentKd = 0

        # Losses
        self.resetTmpLosses()

        # WGAN-GP
        self.config.lambdaGP = lambdaGP

        # Weight on D's output
        self.config.epsilonD = epsilonD

    # used in test time, no backprop
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

    def resetTmpLosses(self):
        r"""
        Reset all internal losses log to zero
        """

        self.trainTmp.lossD = 0
        self.trainTmp.lossG = 0

        self.trainTmp.lossEpsilon = 0.
        self.trainTmp.lossACD = 0.0
        self.trainTmp.lossACG = 0.0
        self.trainTmp.lossGrad = 0.0

    def optimizeParameters(self, input_batch, inputLabels=None):
        r"""
        Update the discrimator D using the given "real" inputs.
        After self.config.kInnerD steps of optimization, the generator G will be
        updated kInnerG times.

        Args:
            input (torch.tensor): input batch of real data

        """

        # Retrieve the input data
        self.real_input = input_batch.to(self.device)

        if self.config.attribKeysOrder:
            self.realLabels = inputLabels.to(self.device)

        n_samples = self.real_input.size()[0]

        # Update the discriminator
        self.resetTmpLosses()
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD.forward(self.real_input)
        lossD = self.lossCriterion.getCriterion(predRealD, True)

        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent)

        if isinstance(predFakeG, list):
            for item in predFakeG:
                item.detach()
        else:
            predFakeG.detach()

        predFakeD = self.netD(predFakeG)
        lossD += self.lossCriterion.getCriterion(predFakeD, False)

        if self.config.lambdaGP > 0:
            tmp = lossD.item()
            lossD += self.getGradientPenalty(self.real_input, predFakeG)
            self.trainTmp.lossGrad += (lossD.item() - tmp)

        if self.config.epsilonD > 0:
            tmp = lossD.item()
            lossD += (predRealD[:, :self.lossCriterion.sizeDecisionLayer]
                      ** 2).sum() * self.config.epsilonD
            self.trainTmp.lossEpsilon += (lossD.item() - tmp)

        if self.config.attribKeysOrder:
            tmp = lossD.item()
            lossD += self.config.weightConditionD \
                * self.getLossACDCriterion(predRealD, self.realLabels) \
                + self.config.weightConditionD * \
                self.getLossACDCriterion(predFakeD, targetRandCat)

            self.trainTmp.lossACD += (lossD.item() - tmp)

        lossD.backward()
        self.trainTmp.lossD += lossD.item()

        self.trainTmp.currentKd += 1
        finiteCheck(self.netD.module.parameters())
        self.optimizerD.step()

        # Can we update the generator ?
        if self.trainTmp.currentKd >= self.config.kInnerD:
            self.trainTmp.currentKd = 0
            self.trainTmp.lossG = 0.0

            # Update kInnerG times the generator
            for iteration in range(self.config.kInnerG):

                self.optimizerG.zero_grad()
                self.optimizerD.zero_grad()

                inputNoise, targetCatNoise = self.buildNoiseData(n_samples)
                predFakeG = self.netG(inputNoise)

                predFakeD = self.netD(predFakeG)
                lossGFake = self.lossCriterion.getCriterion(predFakeD, True)

                self.trainTmp.lossG += lossGFake.item()
                self.auxiliaryLossesGeneration()

                self.trainTmp.lossACG += self.updateLossACGeneration(
                    predFakeD, targetCatNoise)
                lossGFake.backward()

                finiteCheck(self.netG.module.parameters())
                self.optimizerG.step()

            # Update the moving average if relevant
            for p, avg_p in zip(self.netG.module.parameters(),
                                self.avgG.module.parameters()):
                avg_p.mul_(0.999).add_(0.001, p.data)

            self.trainTmp.lossG /= self.config.kInnerG

    def initializeACCriterion(self):
        r"""
        """

        if self.config.weightConditionD != 0 and not self.config.attribKeysOrder:
            raise AttributeError("If the weight on the conditional term isn't "
                                 "null, then a attribute dictionnery should be"
                                 " defined")

        if self.config.weightConditionG != 0 and not self.config.attribKeysOrder:
            raise AttributeError("If the weight on the conditional term isn't \
                                 null, then a attribute dictionnery should be \
                                 defined")

        if self.config.attribKeysOrder is not None:
            self.ACGANCriterion = ACGanCriterion(self.config.attribKeysOrder)
            self.config.categoryVectorDim = self.ACGANCriterion.getInputDim()

    def getLossACDCriterion(self, predD, targetLabel):
        r"""
        Retrieve the loss due to the AC-GAN criterion
        Args:
            predD (tensor): output of the discrimator network
            targetLabel (tensor): target output label (! format)
        Return:
            The loss
        """
        xD = predD[:, self.lossCriterion.sizeDecisionLayer:]
        return self.ACGANCriterion.getLoss(xD, targetLabel)

    def updateLossACGeneration(self, predD, targetCatNoise):
        r"""
        Retrieve the generator's loss due to the the AC-GAN criterion
        Depending on self.config.weightConditionG's sign, the creativity loss
        will be activated.
        Args:
            predD (tensor): output of the discrimator network
            targetLabel (tensor): target output label (! format)
        Return:
            The loss
        """

        if self.config.attribKeysOrder is None:
            return 0

        predFakeD = predD[:, self.lossCriterion.sizeDecisionLayer:]
        loss = self.config.weightConditionG * \
            self.ACGANCriterion.getLoss(predFakeD, targetCatNoise)

        loss.backward(retain_graph=True)
        return loss.item()

    def auxiliaryLossesGeneration(self):
        r"""
        For children classes, additional loss put on the generator.
        """
        return

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

    def buildNoiseData(self, n_samples, sameCriterion=False):
        r"""
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """

        inputLatent = torch.randn(
            n_samples, self.config.noiseVectorDim).to(self.device)

        if self.config.attribKeysOrder:

            if sameCriterion:
                targetRandCat, latentRandCat = \
                    self.ACGANCriterion.buildRandomCriterionTensor(1)
                targetRandCat = targetRandCat.expand(n_samples, -1)
                latentRandCat = latentRandCat.expand(n_samples, -1)
            else:
                targetRandCat, latentRandCat = \
                    self.ACGANCriterion.buildRandomCriterionTensor(n_samples)

            targetRandCat = targetRandCat.to(self.device)
            latentRandCat = latentRandCat.to(self.device)
            inputLatent = torch.cat((inputLatent, latentRandCat), dim=1)

            return inputLatent, targetRandCat

        return inputLatent, None

    def buildNoiseDataWithConstraints(self, n, labels):

        constrainPart = self.ACGANCriterion.generateConstraintsFromVector(
            n, labels)
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
             path,
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

        # Step one : load the configuration
        if loadConfig:
            updateConfig(self.config, in_state['config'])
            if self.config.lossCriterion == 'WGANGP2':
                self.config.lossCriterion = 'WGANGP'
            self.lossCriterion = getattr(
                base_loss_criterions, self.config.lossCriterion)(self.device)
            self.initializeACCriterion()

        # Re-initialize G and D with the loaded configuration
        buildAvG = True

        if loadG:
            self.netG = self.getNetG()
            if finetuning:
                loadPartOfStateDict(
                    self.netG, in_state['netG'], ["formatLayer"])
                self.getOriginalG().initFormatLayer(self.config.latentVectorDim)
            else:
                self.netG.load_state_dict(in_state['netG'])
                if 'avgG' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    self.avgG.module.load_state_dict(in_state['avgG'])
                    buildAvG = False

        if loadD:

            # Possibility to convert a B&W discriminator into a color one
            makeRGBTransfer = False
            if self.config.dimOutput == 3 and in_state['config'].dimOutput == 1:
                self.config.dimOutput = 1
                makeRGBTransfer = True

            self.netD = self.getNetD()

            if finetuning:
                loadPartOfStateDict(
                    self.netD, in_state['netD'], ["decisionLayer"])
                self.getOriginalD().initDecisionLayer(
                    self.lossCriterion.sizeDecisionLayer
                    + self.config.categoryVectorDim)
            else:
                self.netD.load_state_dict(in_state['netD'])

            if makeRGBTransfer:
                self.netD.switch2RGBInput()
                self.config.dimOutput = 3

        elif 'tmp' in in_state.keys():
            self.trainTmp = in_state['tmp']

        self.loadAuxiliaryData(in_state)

        # Don't forget to reset the machinery !
        self.updateSolversDevice(buildAvG)

    def loadAuxiliaryData(self, in_state):
        r"""
        For children classes, in any supplementary data should be loaded from an
        input state dictionary, it should be defined here.
        """
        return

    def getGradientPenalty(self, input, fake):
        r"""
        Build the gradient penalty as described in
        "Improved Training of Wasserstein GANs"

        Args:

            - input (Tensor): batch of real data
            - fake (Tensor): batch of generated data. Must have the same size as
            the input
        """

        batchSize = input.size(0)
        alpha = torch.rand(batchSize, 1)

        alpha = alpha.expand(batchSize, input.nelement() /
                             batchSize).contiguous().view(input.size())
        alpha = alpha.to(self.device)
        interpolates = alpha * input + ((1 - alpha) * fake)

        interpolates = torch.autograd.Variable(
            interpolates, requires_grad=True)

        decisionInterpolate = self.netD(interpolates)[:, 0].sum()

        gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            decisionInterpolate.size()).to(self.device),
                                        create_graph=True, retain_graph=True,
                                        only_inputs=True)

        gradients = gradients[0].view(batchSize, -1)
        gradients = (gradients * gradients).sum(dim=1).sqrt()
        gradient_penalty = (((gradients - 1.0)**2)).sum()

        return gradient_penalty * self.config.lambdaGP

    def gradientDescentOnInput(self,
                               input,
                               featureExtractors,
                               imageTransforms,
                               weights=None,
                               visualizer=None,
                               lambdaD=0.03,
                               nSteps=6000,
                               randomSearch=False,
                               lr=1,
                               outPathSave=None):

        if visualizer is not None:
            visualizer.publishTensors(input, (128, 128))

        # Detect categories
        varNoise = torch.randn((input.size(0),
                                self.config.noiseVectorDim +
                                self.config.categoryVectorDim,
                                1, 1),
                               requires_grad=True, device=self.device)

        optimNoise = optim.Adam([varNoise],
                                betas=[0., 0.99], lr=lr)

        noiseOut = self.test(varNoise, getAvG=True, toCPU=False)

        if visualizer is not None:
            visualizer.publishTensors(noiseOut.cpu(), (128, 128))

        if not isinstance(featureExtractors, list):
            featureExtractors = [featureExtractors]
        if not isinstance(imageTransforms, list):
            imageTransforms = [imageTransforms]

        nExtractors = len(featureExtractors)

        if weights is None:
            weights = [1.0 for i in range(nExtractors)]

        if len(imageTransforms) != nExtractors:
            raise ValueError(
                "The number of image transforms should match the number of \
                feature extractors")
        if len(weights) != nExtractors:
            raise ValueError(
                "The number of weights should match the number of feature\
                 extractors")

        featuresIn = []

        for i in range(nExtractors):

            if len(featureExtractors[i]._modules) > 0:
                featureExtractors[i] = nn.DataParallel(
                    featureExtractors[i]).train().to(self.device)

            imageTransforms[i] = nn.DataParallel(
                imageTransforms[i]).to(self.device)

            featuresIn.append(featureExtractors[i](
                imageTransforms[i](input.to(self.device))).detach())

        lr = 1

        optimalVector = None
        optimalLoss = None

        epochStep = int(nSteps / 3)
        gradientDecay = 0.1

        def resetVar(newVal):
            varNoise = newVal
            optimNoise = optim.Adam([varNoise],
                                    betas=[0., 0.99], lr=lr)

        for iter in range(nSteps):

            optimNoise.zero_grad()
            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()

            if randomSearch:
                varNoise = torch.randn((input.size(0),
                                        self.config.noiseVectorDim +
                                        self.config.categoryVectorDim,
                                        1, 1),
                                       requires_grad=True, device=self.device)

            noiseOut = self.avgG(varNoise)
            sumLoss = 0

            loss = ((varNoise**2).mean(dim=1) - 1)**2
            loss.backward(retain_graph=True)
            sumLoss += loss.item()

            for i in range(nExtractors):
                featureOut = featureExtractors[i](imageTransforms[i](noiseOut))
                diff = ((featuresIn[i] - featureOut)**2)
                loss = weights[i] * diff.mean()
                sumLoss += loss.item()

                if not randomSearch:
                    loss.backward(retain_graph=True)

            loss = -lambdaD * self.netD(noiseOut)[0, 0]
            sumLoss += loss.item()

            if not randomSearch:
                loss.backward()

            sumLoss += loss.item()

            if not randomSearch:
                optimNoise.step()

            if optimalLoss is None or sumLoss < optimalLoss:
                optimalVector = deepcopy(varNoise)
                optimalLoss = sumLoss

            if iter % 100 == 0:
                if visualizer is not None:
                    visualizer.publishTensors(noiseOut.cpu(), (128, 128))

                    if outPathSave is not None:
                        index_str = str(int(iter/100))
                        outPath = os.path.join(outPathSave, index_str + ".jpg")
                        visualizer.saveTensor(
                            noiseOut.cpu(),
                            (noiseOut.size(2), noiseOut.size(3)),
                            outPath)

                print("%d : %f" % (iter, sumLoss))

            if iter % epochStep == (epochStep - 1):
                lr *= gradientDecay
                resetVar(optimalVector)

        varNoise = optimalVector
        output = self.test(varNoise, getAvG=True, toCPU=True).detach()

        if visualizer is not None:
            visualizer.publishTensors(
                output.cpu(), (output.size(2), output.size(3)))

        print("optimal loss %f" % optimalLoss)
        return output, varNoise, optimalLoss
