# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import pickle as pkl

import torch
import torchvision.transforms as Transforms

from ..utils.config import getConfigFromDict, getDictFromConfig, BaseConfig
from ..utils.image_transform import NumpyResize, NumpyToTensor
from ..datasets.attrib_dataset import AttribDataset
from ..datasets.hd5 import H5Dataset


class GANTrainer():
    r"""
    A class managing a progressive GAN training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    def __init__(self,
                 pathdb,
                 useGPU=True,
                 visualisation=None,
                 lossIterEvaluation=200,
                 saveIter=5000,
                 checkPointDir=None,
                 modelLabel="GAN",
                 config=None,
                 pathAttribDict=None,
                 selectedAttributes=None,
                 imagefolderDataset=False,
                 ignoreAttribs=False,
                 pathPartition=None,
                 partitionValue=None):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
            dataset.
            - useGPU (bool): set to True if you want to use the available GPUs
            for the training procedure
            - visualisation (module): if not None, a visualisation module to
            follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
            model's loss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
            (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the
            checkpoints should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary.
            for all the possible options
            - pathAttribDict (string): path to the attribute dictionary giving
                                       the labels of the dataset
            - selectedAttributes (list): if not None, consider only the listed
                                     attributes for labelling
            - imagefolderDataset (bool): set to true if the data are stored in
                                        the fashion of a
                                        torchvision.datasests.ImageFolderDataset
                                        object
            - ignoreAttribs (bool): set to True if the input attrib dict should
                                    only be used as a filter on image's names
            - pathPartition (string): if only a subset of the original dataset
                                      should be used
            - pathValue (string): partition value
        """

        # Parameters
        # Training dataset
        self.path_db = pathdb
        self.pathPartition = pathPartition
        self.partitionValue = partitionValue

        if config is None:
            config = {}

        # Load the training configuration
        self.readTrainConfig(config)

        # Initialize the model
        self.useGPU = useGPU

        if not self.useGPU:
            self.numWorkers = 1

        self.pathAttribDict = pathAttribDict
        self.selectedAttributes = selectedAttributes
        self.imagefolderDataset = imagefolderDataset
        self.modelConfig.attribKeysOrder = None

        if (not ignoreAttribs) and \
                (self.pathAttribDict is not None or self.imagefolderDataset):
            self.modelConfig.attribKeysOrder = self.getDataset(
                0, size=10).getKeyOrders()

            print("AC-GAN classes : ")
            print(self.modelConfig.attribKeysOrder)
            print("")

        # Intern state
        self.runningLoss = {}
        self.startScale = 0
        self.startIter = 0
        self.lossProfile = []

        self.initModel()

        print("%d images detected" % int(len(self.getDataset(0, size=10))))

        # Visualization ?
        self.visualisation = visualisation
        self.tokenWindowFake = None
        self.tokenWindowFakeSmooth = None
        self.tokenWindowReal = None
        self.tokenWindowLosses = None
        self.refVectorPath = None

        self.nDataVisualization = 16
        self.refVectorVisualization, self.refVectorLabels = \
            self.model.buildNoiseData(self.nDataVisualization)

        # Checkpoints ?
        self.checkPointDir = checkPointDir
        self.modelLabel = modelLabel
        self.saveIter = saveIter
        self.pathLossLog = None


        if self.checkPointDir is not None:
            self.pathLossLog = os.path.abspath(os.path.join(self.checkPointDir,
                                                            self.modelLabel
                                                            + '_losses.pkl'))
            self.pathRefVector = os.path.abspath(os.path.join(self.checkPointDir,
                                                              self.modelLabel
                                                              + '_refVectors.pt'))

        # Loss printing
        self.lossIterEvaluation = lossIterEvaluation

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        pass

    def updateRunningLosses(self, allLosses):

        for name, value in allLosses.items():

            if name not in self.runningLoss:
                self.runningLoss[name] = [0, 0]

            self.runningLoss[name][0]+= value
            self.runningLoss[name][1]+=1

    def resetRunningLosses(self):

        self.runningLoss = {}

    def updateLossProfile(self, iter):

        nPrevIter = len(self.lossProfile[-1]["iter"])
        self.lossProfile[-1]["iter"].append(iter)

        newKeys = set(self.runningLoss.keys())
        existingKeys = set(self.lossProfile[-1].keys())

        toComplete = existingKeys - newKeys

        for item in newKeys:

            if item not in existingKeys:
                self.lossProfile[-1][item] = [None for x in range(nPrevIter)]

            value, stack = self.runningLoss[item]
            self.lossProfile[-1][item].append(value /float(stack))

        for item in toComplete:
            if item in ["scale", "iter"]:
                continue
            self.lossProfile[-1][item].append(None)

    def readTrainConfig(self, config):
        r"""
        Load a permanent configuration describing a models. The variables
        described in this file are constant through the training.
        """
        self.modelConfig = BaseConfig()
        getConfigFromDict(self.modelConfig, config, self.getDefaultConfig())

    def loadSavedTraining(self,
                          pathModel,
                          pathTrainConfig,
                          pathTmpConfig,
                          loadGOnly=False,
                          loadDOnly=False,
                          finetune=False):
        r"""
        Load a given checkpoint.

        Args:

            - pathModel (string): path to the file containing the model
                                 structure (.pt)
            - pathTrainConfig (string): path to the reference configuration
                                        file of the training. WARNING: this
                                        file must be compatible with the one
                                        pointed by pathModel
            - pathTmpConfig (string): path to the temporary file describing the
                                      state of the training when the checkpoint
                                      was saved. WARNING: this file must be
                                      compatible with the one pointed by
                                      pathModel
        """

        # Load the temp configuration
        tmpPathLossLog = None
        tmpConfig = {}

        if pathTmpConfig is not None:
            tmpConfig = json.load(open(pathTmpConfig, 'rb'))
            self.startScale = tmpConfig["scale"]
            self.startIter = tmpConfig["iter"]
            self.runningLoss = tmpConfig.get("runningLoss", {})

            tmpPathLossLog = tmpConfig.get("lossLog", None)

        if tmpPathLossLog is None:
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        elif not os.path.isfile(tmpPathLossLog):
            print("WARNING : couldn't find the loss logs at " +
                  tmpPathLossLog + " resetting the losses")
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        else:
            self.lossProfile = pkl.load(open(tmpPathLossLog, 'rb'))
            self.lossProfile = self.lossProfile[:(self.startScale + 1)]

            if self.lossProfile[-1]["iter"][-1] > self.startIter:
                indexStop = next(x[0] for x in enumerate(self.lossProfile[-1]["iter"])
                                 if x[1] > self.startIter)
                self.lossProfile[-1]["iter"] = self.lossProfile[-1]["iter"][:indexStop]

                for item in self.lossProfile[-1]:
                    if isinstance(self.lossProfile[-1][item], list):
                        self.lossProfile[-1][item] = \
                            self.lossProfile[-1][item][:indexStop]

        # Read the training configuration
        if not finetune:
            trainConfig = json.load(open(pathTrainConfig, 'rb'))
            self.readTrainConfig(trainConfig)

        # Re-initialize the model
        self.initModel()
        self.model.load(pathModel,
                        loadG=not loadDOnly,
                        loadD=not loadGOnly,
                        finetuning=finetune)

        # Build retrieve the reference vectors
        self.refVectorPath = tmpConfig.get("refVectors", None)
        if self.refVectorPath is None:
            self.refVectorVisualization, self.refVectorLabels = \
                self.model.buildNoiseData(self.nDataVisualization)
        elif not os.path.isfile(self.refVectorPath):
            print("WARNING : no file found at " + self.refVectorPath
                  + " building new reference vectors")
            self.refVectorVisualization, self.refVectorLabels = \
                self.model.buildNoiseData(self.nDataVisualization)
        else:
            self.refVectorVisualization = torch.load(
                open(self.refVectorPath, 'rb'))

    def getDefaultConfig(self):
        pass

    def resetVisualization(self, nDataVisualization):

        self.nDataVisualization = nDataVisualization
        self.refVectorVisualization, self.refVectorLabels = \
            self.model.buildNoiseData(self.nDataVisualization)

    def saveBaseConfig(self, outPath):
        r"""
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(
            self.modelConfig, self.getDefaultConfig())

        if "alphaJumpMode" in outConfig:
            if outConfig["alphaJumpMode"] == "linear":

                outConfig.pop("iterAlphaJump", None)
                outConfig.pop("alphaJumpVals", None)

        with open(outPath, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

    def saveCheckpoint(self, outDir, outLabel, scale, iter):
        r"""
        Save a checkpoint at the given directory. Please not that the basic
        configuration won't be saved.

        This function produces 2 files:
        outDir/outLabel_tmp_config.json -> temporary config
        outDir/outLabel -> networks' weights

        And update the two followings:
        outDir/outLabel_losses.pkl -> losses util the last registered iteration
        outDir/outLabel_refVectors.pt -> reference vectors for visualization
        """
        pathModel = os.path.join(outDir, outLabel + ".pt")
        self.model.save(pathModel)

        # Tmp Configuration
        pathTmpConfig = os.path.join(outDir, outLabel + "_tmp_config.json")
        outConfig = {'scale': scale,
                     'iter': iter,
                     'lossLog': self.pathLossLog,
                     'refVectors': self.pathRefVector,
                     'runningLoss': self.runningLoss}

        # Save the reference vectors
        torch.save(self.refVectorVisualization, open(self.pathRefVector, 'wb'))

        with open(pathTmpConfig, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

        if self.pathLossLog is None:
            raise AttributeError("Logging mode disabled")

        if self.pathLossLog is not None:
            pkl.dump(self.lossProfile, open(self.pathLossLog, 'wb'))

        if self.visualisation is not None:
            ref_g = self.model.test(self.refVectorVisualization)
            imgSize = max(128, ref_g.size()[2])
            self.visualisation.saveTensor(ref_g, (imgSize, imgSize),
                                          os.path.join(outDir, outLabel + '.jpg'))

            ref_g_smooth = self.model.test(self.refVectorVisualization, True)
            self.visualisation.saveTensor(ref_g_smooth, (imgSize, imgSize),
                                          os.path.join(outDir, outLabel + '_avg.jpg'))

    def sendToVisualization(self, refVectorReal, scale, label=None):
        r"""
        Send the images generated from some reference latent vectors and a
        bunch of real examples from the dataset to the visualisation tool.
        """
        imgSize = max(128, refVectorReal.size()[2])
        envLabel = self.modelLabel + "_training"

        if label is None:
            label = self.modelLabel
        else:
            self.visualisation.publishTensors(refVectorReal,
                                              (imgSize, imgSize),
                                              label + " real",
                                              env=envLabel)

        ref_g_smooth = self.model.test(self.refVectorVisualization, True)
        self.tokenWindowFakeSmooth = \
            self.visualisation.publishTensors(ref_g_smooth,
                                              (imgSize, imgSize),
                                              label + " smooth",
                                              self.tokenWindowFakeSmooth,
                                              env=envLabel)

        ref_g = self.model.test(self.refVectorVisualization, False)

        self.tokenWindowFake = \
            self.visualisation.publishTensors(ref_g,
                                              (imgSize, imgSize),
                                              label + " fake",
                                              self.tokenWindowFake,
                                              env=envLabel)
        self.tokenWindowReal = \
            self.visualisation.publishTensors(refVectorReal,
                                              (imgSize, imgSize),
                                              label + " real",
                                              self.tokenWindowReal,
                                              env=envLabel)
        self.tokenWindowLosses = \
            self.visualisation.publishLoss(self.lossProfile[scale],
                                           self.modelLabel,
                                           self.tokenWindowLosses,
                                           env=envLabel)

    def getDBLoader(self, scale):
        r"""
        Load the training dataset for the given scale.

        Args:

            - scale (int): scale at which we are working

        Returns:

            A dataset with properly resized inputs.
        """
        dataset = self.getDataset(scale)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.modelConfig.miniBatchSize,
                                           shuffle=True, num_workers=self.model.n_devices)

    def getDataset(self, scale, size=None):

        if size is None:
            size = self.model.getSize()

        isH5 = os.path.splitext(self.path_db)[1] == ".h5"

        print("size", size)
        transformList = [NumpyResize(size),
                         NumpyToTensor(),
                         Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        if self.modelConfig.dimOutput == 1:
            transformList = [Transforms.Grayscale(1)] + transformList

        transform = Transforms.Compose(transformList)

        if isH5:
            return H5Dataset(self.path_db,
                             partition_path=self.pathPartition,
                             partition_value=self.partitionValue,
                             specificAttrib=self.selectedAttributes,
                             stats_file=self.pathAttribDict,
                             transform=transform)

        return AttribDataset(self.path_db,
                             transform=transform,
                             attribDictPath=self.pathAttribDict,
                             specificAttrib=self.selectedAttributes,
                             mimicImageFolder=self.imagefolderDataset)

    def inScaleUpdate(self, iter, scale, inputs_real):
        return inputs_real

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     shiftIter=0,
                     maxIter=-1):
        r"""
        Train the model on one epoch.

        Args:

            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when
                               looking for the next update of the alpha
                               coefficient
            - maxIter (int): if > 0, iteration at which the training should stop

        Returns:

            True if the training went smoothly
            False if a diverging behavior was detected and the training had to
            be stopped
        """

        i = shiftIter

        for item, data in enumerate(dbLoader, 0):

            inputs_real = data[0]
            labels = data[1]

            if inputs_real.size()[0] < self.modelConfig.miniBatchSize:
                continue

            # Additionnal updates inside a scale
            inputs_real = self.inScaleUpdate(i, scale, inputs_real)

            if len(data) > 2:
                mask = data[2]
                allLosses = self.model.optimizeParameters(
                    inputs_real, inputLabels=labels, inputMasks=mask)
            else:
                allLosses = self.model.optimizeParameters(inputs_real,
                                                          inputLabels=labels)

            self.updateRunningLosses(allLosses)

            i += 1

            # Regular evaluation
            if i % self.lossIterEvaluation == 0:

                # Reinitialize the losses
                self.updateLossProfile(i)

                print('[%d : %6d] loss G : %.3f loss D : %.3f' % (scale, i,
                      self.lossProfile[-1]["lossG"][-1],
                      self.lossProfile[-1]["lossD"][-1]))

                self.resetRunningLosses()

                if self.visualisation is not None:
                    self.sendToVisualization(inputs_real, scale)

            if self.checkPointDir is not None:
                if i % self.saveIter == 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, i))
                    self.saveCheckpoint(self.checkPointDir,
                                        labelSave, scale, i)

            if i == maxIter:
                return True

        return True

    def train(self):
        pass
