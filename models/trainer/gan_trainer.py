import os
import json
import logging
import pickle as pkl

import torch
import torchvision
import torchvision.transforms as Transforms

from ..utils.config import getConfigFromDict, getDictFromConfig, BaseConfig
from ..utils.image_transform import NumpyFlip, NumpyResize, NumpyToTensor
from ..datasets.attrib_dataset import AttribDataset
from ..datasets.hd5 import H5Dataset

class GANTrainer():
    r"""
    A class managing a progressive GAN training. Logs, chekpoints, visualization,
    and number iterations are managed here.
    """
    def __init__(self,
                 pathdb,
                 useGPU = True,
                 visualisation = None,
                 lossIterEvaluation = 200,
                 saveIter = 5000,
                 checkPointDir = None,
                 modelLabel = "GAN",
                 config = None,
                 pathAttribDict = None,
                 specificAttrib=None,
                 imagefolderDataset = False,
                 celebaHQDB = False,
                 ignoreAttribs = False,
                 pathDBMask = None,
                 pathPartition = None,
                 partitionValue = None):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image dataset
            TODO: now the dataset is loaded using the torchvision.datasets.ImageFolder
            function which requires each image to have a label. Change that.
            - useGPU (bool): set to True if you want to use the available GPUs
            for the training procedure
            - visualisation (module): if not None, a visualisation module to
            follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the model's
            loss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
            (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the checkpoints
            should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary.
            for all the possible options
            - imagefolderDataset (bool): set to true if the data are stored in the fashion of a
                                        torchvision.datasests.ImageFolderDataset object
            - celebaHQDB (bool): set to true if the input images are in the .npy
                                 format
            - configScheduler (dictionary): if the model configurartion should
                                            be updated during the training
            - ignoreAttribs (bool): set to True if the input attrib dict should
                                    only be used as a filter on image's names
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

        self.celebaHQDB = celebaHQDB

        self.pathAttribDict = pathAttribDict
        self.specificAttrib = specificAttrib
        self.imagefolderDataset = imagefolderDataset
        self.modelConfig.attribKeysOrder = None

        self.pathDBMask = pathDBMask

        if (not ignoreAttribs) and (self.pathAttribDict is not None or self.imagefolderDataset):
            self.modelConfig.attribKeysOrder = self.getDataset(0, size = 10).getKeyOrders(self.modelConfig.equalizeLabels)

            print("AC-GAN classes : ")
            print(self.modelConfig.attribKeysOrder)
            print("")

            self.modelConfig.maskExtraction = pathDBMask is not None

        self.initModel()

        print("%d images detected" % int(len(self.getDataset(0, size = 10))))

        # Visualization ?
        self.visualisation = visualisation
        self.tokenWindowFake = None
        self.tokenWindowFakeSmooth = None
        self.tokenWindowReal = None
        self.tokenWindowLosses = None
        self.refVectorPath = None

        self.nDataVisualization = 16
        self.refVectorVisualization, self.refVectorLabels = self.model.buildNoiseData(self.nDataVisualization)

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

        # Intern state
        self.runningLossG = 0
        self.runningLossD = 0
        self.runningLossGrad = 0

        self.runningLoss = {"stack":0}

        self.startScale = 0
        self.startIter = 0

        self.lossProfile = []

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        pass

    def updateRunningLosses(self):

        self.runningLoss["stack"] += 1

        for name, value in vars(self.model.trainTmp).items():

            if name[:4] != "loss":
                continue

            if name not in self.runningLoss:
                self.runningLoss[name] = 0

            self.runningLoss[name] += value

    def resetRunningLosses(self):

        for item in self.runningLoss:
            self.runningLoss[item] = 0

    def updateLossProfile(self):

        stack = self.runningLoss["stack"]

        for item, value in self.runningLoss.items():
            if item =="stack":
                continue
            if item not in self.lossProfile[-1]:
                self.lossProfile[-1][item] = []

            self.lossProfile[-1][item].append(value / float(stack))

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
                          loadGOnly = False,
                          loadDOnly = False,
                          finetune = False):
        r"""
        Load a given checkpoint.

        Args:

            - pathModel (string): path to the file containing the model structure
                                (.pt)
            - pathTrainConfig (string): path to the reference configuration file
                                        of the training. WARNING: this file must be
                                        compatible with the one pointed by
                                        pathModel
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
            self.runningLossG = tmpConfig["runningLossG"]
            self.runningLossD = tmpConfig["runningLossD"]
            self.startScale = tmpConfig["scale"]
            self.startIter = tmpConfig["iter"]

            tmpPathLossLog = tmpConfig.get("lossLog", None)

        if tmpPathLossLog is None:
            self.lossProfile = [{"iter" : [], "G" : [], "D" : [], "scale" : self.startScale}]
        elif not os.path.isfile(tmpPathLossLog):
            print("WARNING : couldn't find the loss logs at " +
                 tmpPathLossLog + " resetting the losses")
            self.lossProfile = [{"iter" : [], "G" : [], "D" : [], "scale" : self.startScale}]
        else:
            self.lossProfile = pkl.load(open(tmpPathLossLog, 'rb'))
            self.lossProfile = self.lossProfile[:(self.startScale +1)]

            if self.lossProfile[-1]["iter"][-1] > self.startIter:
                indexStop = next(x[0] for x in enumerate(self.lossProfile[-1]["iter"])
                            if x[1] > self.startIter)
                self.lossProfile[-1]["iter"] = self.lossProfile[-1]["iter"][:indexStop]
                self.lossProfile[-1]["G"] = self.lossProfile[-1]["G"][:indexStop]
                self.lossProfile[-1]["D"] = self.lossProfile[-1]["D"][:indexStop]

        # Read the training configuration
        if not finetune:
            trainConfig = json.load(open(pathTrainConfig, 'rb'))
            self.readTrainConfig(trainConfig)

        # Re-initialize the model
        self.initModel()
        self.model.load(pathModel,
                        loadG = not loadDOnly,
                        loadD = not loadGOnly,
                        finetuning = finetune)

        # Build retrieve the reference vectors
        self.refVectorPath = tmpConfig.get("refVectors", None)
        if self.refVectorPath is None:
            self.refVectorVisualization, self.refVectorLabels = self.model.buildNoiseData(self.nDataVisualization)
        elif not os.path.isfile(self.refVectorPath):
            print("WARNING : no file found at " + self.refVectorPath
                    + " building new reference vectors")
            self.refVectorVisualization, self.refVectorLabels = self.model.buildNoiseData(self.nDataVisualization)
        else:
            self.refVectorVisualization = torch.load(open(self.refVectorPath, 'rb'))

    def getDefaultConfig(self):
        pass

    def resetVisualization(self, nDataVisualization):

        self.nDataVisualization = nDataVisualization
        self.refVectorVisualization, self.refVectorLabels = self.model.buildNoiseData(self.nDataVisualization)

    def saveBaseConfig(self, outPath):
        r"""
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(self.modelConfig, self.getDefaultConfig())

        if "alphaJumpMode" in outConfig:
            if outConfig["alphaJumpMode"] == "linear":

                outConfig.pop("iterAlphaJump", None)
                outConfig.pop("alphaJumpVals", None)

        with open(outPath, 'w') as fp:
            json.dump(outConfig, fp,indent=4)

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
        outConfig = {'runningLossG' : self.runningLossG,
                     'runningLossD' : self.runningLossD,
                     'scale' : scale,
                     'iter' : iter,
                     'lossLog': self.pathLossLog,
                     'refVectors': self.pathRefVector}

        #Save the reference vectors
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
                                          os.path.join(outDir, outLabel +'.jpg'))

            ref_g_smooth = self.model.test(self.refVectorVisualization, True)
            self.visualisation.saveTensor(ref_g_smooth, (imgSize, imgSize),
                                          os.path.join(outDir, outLabel +'_avg.jpg'))

    def sendToVisualization(self, refVectorReal, scale, label = None):
        r"""
        Send the images generated from some reference latent vectors and a bunch
        of real examples from the dataset to the visualisation tool.
        """
        imgSize = max(128, refVectorReal.size()[2])

        if label is None:
            label = self.modelLabel
        else:
            self.visualisation.publishTensors(refVectorReal,
                                             (imgSize, imgSize),
                                             label + " real",
                                             env = self.modelLabel)

        ref_g_smooth = self.model.test(self.refVectorVisualization, True)
        self.tokenWindowFakeSmooth = self.visualisation.publishTensors(ref_g_smooth, (imgSize, imgSize),
                                                            label + " smooth",
                                                            self.tokenWindowFakeSmooth,
                                                            env = self.modelLabel)

        ref_g = self.model.test(self.refVectorVisualization, False)

        self.tokenWindowFake = self.visualisation.publishTensors(ref_g, (imgSize, imgSize),
                                                            label + " fake",
                                                            self.tokenWindowFake,
                                                            env = self.modelLabel)
        self.tokenWindowReal = self.visualisation.publishTensors(refVectorReal,
                                                            (imgSize, imgSize),
                                                            label + " real",
                                                            self.tokenWindowReal,
                                                            env = self.modelLabel)
        self.tokenWindowLosses = self.visualisation.publishLoss(self.lossProfile[scale],
                                                                self.modelLabel,
                                                                self.tokenWindowLosses,
                                                                env = self.modelLabel)

    def getDBLoader(self, scale):
        r"""
        Load the training dataset for the given scale.

        Args:

            - scale (int): scale at which we are working

        Returns:

            A dataset with properly resized inputs.
        """
        dataset = self.getDataset(scale)
        print(self.modelConfig.miniBatchSize, self.model.n_devices)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.modelConfig.miniBatchSize,
                                           shuffle=True, num_workers= self.model.n_devices)

    def getDataset(self, scale, size = None):

        if size is None:
            size = self.model.getSize()

        isH5 = os.path.splitext(self.path_db)[1] == ".h5"

        transformList = [NumpyResize(size),
                        # NumpyFlip(),
                         NumpyToTensor(),
                         Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        if self.modelConfig.dimOutput == 1:
            transformList = [Transforms.Grayscale(1)] + transformList

        transform = Transforms.Compose(transformList)

        if isH5:
            return H5Dataset(self.path_db,
                             partition_path = self.pathPartition,
                             partition_value = self.partitionValue,
                             specificAttrib = self.specificAttrib,
                             stats_file = self.pathAttribDict,
                             transform = transform,
                             pathDBMask = self.pathDBMask)

        return AttribDataset(self.path_db, transform = transform,
                             attribDictPath = self.pathAttribDict,
                             specificAttrib=self.specificAttrib,
                             mimicImageFolder = self.imagefolderDataset,
                             pathMask = self.pathDBMask)

    def inScaleUpdate(self, iter, scale, inputs_real):
        return inputs_real

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     shiftIter = 0,
                     maxIter=-1):
        r"""
        Train the model on one epoch.

        Args:

            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when looking
                               for the next update of the alpha coefficient
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
                self.model.optimizeParameters(inputs_real, inputLabels = labels, inputMasks = mask)
            else:
                self.model.optimizeParameters(inputs_real, inputLabels = labels)

            self.runningLossG+= self.model.trainTmp.lossG
            self.runningLossD+= self.model.trainTmp.lossD

            self.updateRunningLosses()

            i+=1

            # Regular evaluation
            if i % self.lossIterEvaluation == 0:
                lossG = self.runningLossG / self.lossIterEvaluation
                lossD = self.runningLossD / self.lossIterEvaluation
                print('[%d : %6d] loss G : %.3f loss D : %.3f' %
                      (scale, i, lossG, lossD))

                self.lossProfile[-1]["G"].append(lossG)
                self.lossProfile[-1]["D"].append(lossD)
                self.lossProfile[-1]["iter"].append(i)

                if self.visualisation is not None:
                    self.sendToVisualization(inputs_real, scale)

                # Reinitialize the losses
                self.runningLossG = 0.0
                self.runningLossD = 0.0

                self.updateLossProfile()
                self.resetRunningLosses()

            if self.checkPointDir is not None:
                if i% self.saveIter == 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, i))
                    self.saveCheckpoint(self.checkPointDir,
                                        labelSave, scale, i)

            if i == maxIter:
                return True

        return True

    def train(self):
        pass
