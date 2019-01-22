import os
import json
import math
import pickle as pkl
from copy import deepcopy

import random

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as Transforms
import numpy as np

from .datasets.attrib_dataset import pil_loader
from .utils.utils import printProgressBar
from .datasets.hd5 import H5Dataset

PI = 3.14159265359


class GANVisualizer():

    def __init__(self,
                 pathGan,
                 pathConfig,
                 ganType,
                 visualizer):

        with open(pathConfig, 'rb') as file:
            self.config = json.load(file)

        # TODO : update me
        self.model = ganType(useGPU=True,
                             storeAVG=True,
                             **self.config)

        self.model.load(pathGan)

        self.visualizer = visualizer
        self.keyShift = None

        self.buildKeyShift()

    def buildKeyShift(self):

        if self.model.config.attribKeysOrder is None:
            return

        self.keyShift = {f: {}
                         for f in self.model.config.attribKeysOrder.keys()}

        for f in self.keyShift:

            order = self.model.config.attribKeysOrder[f]["order"]

            baseShift = sum([len(self.model.config.attribKeysOrder[f]["values"])
                             for f in self.model.config.attribKeysOrder
                             if self.model.config.attribKeysOrder[f]["order"] < order])
            for index, item in enumerate(self.model.config.attribKeysOrder[f]["values"]):
                self.keyShift[f][item] = baseShift + index

    def exploreParameter(self, parameterIndex, size, nsteps=16):

        nsteps = max(nsteps, 2)
        noiseData, _ = self.model.buildNoiseData(1)

        groupNoise = [noiseData.clone() for x in range(nsteps)]
        noiseData = torch.cat(groupNoise, dim=0)
        stepVal = (50) / (float(nsteps) - 2.0)

        currVal = -25

        for i in range(nsteps):

            refVal = currVal
            #noiseData[i] *=  math.sin(currVal)
            noiseData[i][parameterIndex] = refVal

            currVal += stepVal

        outImg = self.model.test(noiseData, getAvG=True)
        outSize = (size, size)

        token = self.visualizer.publishTensors(
            outImg, outSize,
            caption="Parameter %d for -1 to 1" % parameterIndex,
            env="visual")

        self.visualizer.publishTensors(
            outImg, outSize,
            caption="Parameter %d for -1 to 1" % parameterIndex,
            env="visual", window_token=token)

    def exportVisualization(self,
                            path,
                            nVisual=128,
                            maxBatchSize=128,
                            export_mask=False):

        num_device = self.model.n_devices
        size = self.model.getSize()

        maxBatchSize = max(1, int(256 / math.log(size, 2)))
        remaining = nVisual
        out = []

        outTexture, outShape = [], []

        while remaining > 0:

            currBatch = min(remaining, maxBatchSize)
            noiseData, _ = self.model.buildNoiseData(currBatch)
            img = self.model.test(noiseData, getAvG=True)
            out.append(img)

            if export_mask:
                try:
                    _, shape, texture = self.model.getDetailledOutput(
                        noiseData)
                    outTexture.append(texture)
                    outShape.append(shape)
                except AttributeError:
                    print("WARNING, no mask available for this model")

            remaining -= currBatch

        toSave = torch.cat(out, dim=0)
        self.visualizer.saveTensor(
            toSave, (toSave.size()[2], toSave.size()[3]), path)

        if len(outTexture) > 0:
            toSave = torch.cat(outTexture, dim=0)
            pathTexture = os.path.splitext(path)[0] + "_texture.png"
            self.visualizer.saveTensor(
                toSave, (toSave.size()[2], toSave.size()[3]), pathTexture)

            toSave = torch.cat(outShape, dim=0)
            pathShape = os.path.splitext(path)[0] + "_shape.png"
            self.visualizer.saveTensor(
                toSave, (toSave.size()[2], toSave.size()[3]), pathShape)

    def showEqualized(self, size):

        noiseData, _ = self.model.buildNoiseData(1)
        eqVector = torch.zeros(noiseData.size()) + 1.0

        outNoise = self.model.test(noiseData)
        outEq = self.model.test(eqVector)

        outSize = (size, size)

        self.visualizer.publishTensors(
            outEq, outSize,
            caption="Unit vector",
            env="visual")

    def generateSphere(self, parameterIndex, nSteps, size):

        M = self.model.config.latentVectorDim
        N = self.model.config.noiseVectorDim

        random.seed()
        angles = [random.random() for x in range(N)]

        nSteps = max(nSteps, 2)

        inputs = torch.zeros(nSteps, N, 1, 1)

        def setSphericalCoords(angles, iStart, output):

            output[iStart] = math.cos(angles[iStart])
            sinProd = math.sin(angles[iStart])

            for i in range(iStart + 1, N):

                output[i] = sinProd * math.cos(angles[i])
                sinProd *= math.sin(angles[i])

            for i in range(iStart):

                sinProd *= math.sin(angles[i])
                output[i] = sinProd * math.cos(angles[i])

            output[iStart - 1] = sinProd * math.sin(angles[iStart - 1])

        angles[parameterIndex] = 0
        currVal = 0

        step = 2.0 * PI / (float(nSteps) - 2.0)

        for i in range(nSteps):

            setSphericalCoords(angles, parameterIndex, inputs[i])
            currVal += step
            angles[parameterIndex] = currVal

        outSize = (size, size)
        cond = torch.zeros(nSteps, M-N, 1, 1) + 1

        inputs = torch.cat((inputs, cond), dim=1)
        outImg = self.model.test(inputs, getAvG=True)

        self.visualizer.publishTensors(
            outImg, outSize,
            caption="Rotation, angle %d" % parameterIndex,
            env="visual")

    def generateImagesFomConstraints(self, nImages, constraints, env="visual"):

        input = self.model.buildNoiseDataWithConstraints(nImages, constraints)
        outImg = self.model.test(input, getAvG=True)

        outSize = (outImg.size()[2], outImg.size()[3])
        self.visualizer.publishTensors(
            outImg, outSize,
            caption="test",
            env=env)

    def generateRandomConstraints(self):

        output = torch.zeros(nImages, M-N)

        for member, values in self.keyShift.item():

            l = len(values)

    def plotLosses(self, pathLoss, name="Data", clear=True):

        with open(pathLoss, 'rb') as file:
            lossData = pkl.load(file)

        nScales = len(lossData)

        for scale in range(nScales):

            locName = name + ("_s%d" % scale)

            if clear:
                self.visualizer.delete_env(locName)

            locIter = lossData[scale]["iter"]

            for plotName, plotData in lossData[scale].items():

                if plotName == "iter":
                    continue
                if not isinstance(plotData, list):
                    continue
                if len(plotData) == 0:
                    continue

                self.visualizer.publishLinePlot([(plotName, plotData)], locIter,
                                                name=plotName,
                                                env=locName)

    def saveInterpolation(self, N, vectorStart, vectorEnd, pathOut):

        sizeStep = 1.0 / (N - 1)
        pathOut = os.path.splitext(pathOut)[0]

        vectorStart = vectorStart.view(1, -1, 1, 1)
        vectorEnd = vectorEnd.view(1, -1, 1, 1)

        nZeros = int(math.log10(N) + 1)
        print(nZeros)

        for i in range(N):
            path = pathOut + str(i).zfill(nZeros) + ".jpg"
            t = i * sizeStep
            vector = (1 - t) * vectorStart + t * vectorEnd

            outImg = self.model.test(vector, getAvG=True, toCPU=True)
            self.visualizer.saveTensor(
                outImg, (outImg.size(2), outImg.size(3)), path)

    def visualizeNN(self, N, k, featureExtractor, imgTransform, nnSearch, names, pathDB):

        batchSize = 16
        nImages = 0

        vectorOut = []

        size = self.model.getSize()

        transform = Transforms.Compose([Transforms.Resize((size, size)),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = None

        if os.path.splitext(pathDB)[1] == ".h5":
            dataset = H5Dataset(pathDB,
                                transform=Transforms.Compose([Transforms.ToTensor(),
                                                              Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        while nImages < N:

            noiseData, _ = self.model.buildNoiseData(batchSize)
            imgOut = self.model.test(
                noiseData, getAvG=True, toCPU=False).detach()

            features = featureExtractor(imgTransform(imgOut)).detach().view(
                imgOut.size(0), -1).cpu().numpy()
            distances, indexes = nnSearch.query(features, k)
            nImages += batchSize

            for p in range(N):

                vectorOut.append(imgOut[p].cpu().view(
                    1, imgOut.size(1), imgOut.size(2), imgOut.size(3)))
                for ki in range(k):

                    i = indexes[p][ki]
                    if dataset is None:
                        path = os.path.join(pathDB, names[i])
                        imgSource = transform(pil_loader(path))
                        imgSource = imgSource.view(1, imgSource.size(
                            0), imgSource.size(1), imgSource.size(2))

                    else:
                        imgSource, _ = dataset[names[i]]
                        imgSource = imgSource.view(1, imgSource.size(
                            0), imgSource.size(1), imgSource.size(2))
                        imgSource = F.upsample(
                            imgSource, size=(size, size), mode='bilinear')

                    vectorOut.append(imgSource)

        vectorOut = torch.cat(vectorOut, dim=0)
        self.visualizer.publishTensors(vectorOut, (224, 224), nrow=k + 1)

    def exportNN(self, N, k, featureExtractor, imgTransform, nnSearch):

        batchSize = 16
        nImages = 0

        vectorOut = np.zeros(k)

        print("Computing the NN metric")
        while nImages < N:

            printProgressBar(nImages, N)

            noiseData, _ = self.model.buildNoiseData(batchSize)
            imgOut = self.model.test(
                noiseData, getAvG=True, toCPU=False).detach()

            features = featureExtractor(imgTransform(imgOut)).detach().view(
                imgOut.size(0), -1).cpu().numpy()
            distances = nnSearch.query(features, k)[0]

            vectorOut += distances.sum(axis=0)
            nImages += batchSize

        printProgressBar(N, N)

        vectorOut /= nImages
        return vectorOut
