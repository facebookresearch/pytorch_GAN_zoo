# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import math
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms
import numpy as np

from .utils.image_transform import NumpyResize, NumpyToTensor
from .datasets.attrib_dataset import pil_loader
from .utils.utils import printProgressBar
from .datasets.hd5 import H5Dataset


class GANVisualizer():
    r"""
    Several tools to export GAN generations
    """

    def __init__(self,
                 pathGan,
                 pathConfig,
                 ganType,
                 visualizer):
        r"""
        Args
            pathGan (string): path to the GAN to load
            pathConfig (string): path to the GAN configuration
            ganType (BaseGANClass): type of GAn to load
            visualizer (visualizer class): either visualizer or np_visualizer
        """

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
        r"""
        Inilialize the labels shift for labelled models
        """

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

    def exportVisualization(self,
                            path,
                            nVisual=128,
                            export_mask=False):
        r"""
        Save an image gathering sevral generations

        Args:
            path (string): output path of the image
            nVisual (int): number of generation to build
            export_mask (bool): for decoupled model, export the mask as well
                                as the full output
        """

        size = self.model.getSize()[0]
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

    def exportDB(self, path, nItems):
        r"""
        Save dataset of fake generations

        Args:
            path (string): output path of the dataset
            nItems (int): number of generation to build
        """

        size = self.model.getSize()
        maxBatchSize = max(1, int(256 / math.log(size[0], 2)))
        remaining = nItems

        index = 0

        if not os.path.isdir(path):
            os.mkdir(path)

        while remaining > 0:
            currBatch = min(remaining, maxBatchSize)
            noiseData, _ = self.model.buildNoiseData(currBatch)
            img = self.model.test(noiseData, getAvG=True, toCPU=True)

            for i in range(currBatch):
                imgPath = os.path.join(path, "gen_" + str(index) + ".jpg")
                self.visualizer.saveTensor(img[i].view(1, 3, size[0], size[1]),
                                           size, imgPath)
                index += 1

            remaining -= currBatch

    def generateImagesFomConstraints(self,
                                     nImages,
                                     constraints,
                                     env="visual",
                                     path=None):
        r"""
        Given label constraints, generate a set of images.

        Args:
            nImages (int): number of images to generate
            constraints (dict): set of constraints in the form of
                                {attribute:label}. For example

                                {"Gender": "Man",
                                "Color": blue}
            env (string): visdom only, visdom environement where the
                          generations should be exported
            path (string): if not None. Path wher the generations should be
                           saved
        """

        input = self.model.buildNoiseDataWithConstraints(nImages, constraints)
        outImg = self.model.test(input, getAvG=True)

        outSize = (outImg.size()[2], outImg.size()[3])
        self.visualizer.publishTensors(
            outImg, outSize,
            caption="test",
            env=env)

        if path is not None:
            self.visualizer.saveTensor(outImg, outSize, path)

    def plotLosses(self, pathLoss, name="Data", clear=True):
        r"""
        Plot some losses in visdom

        Args:

            pathLoss (string): path to the pickle file where the loss are
                               stored
            name (string): model name
            clear (bool): if True clear the visdom environement before plotting
        """

        with open(pathLoss, 'rb') as file:
            lossData = pkl.load(file)

        nScales = len(lossData)

        for scale in range(nScales):

            locName = name + ("_s%d" % scale)

            if clear:
                self.visualizer.delete_env(locName)

            self.visualizer.publishLoss(lossData[scale],
                                        locName,
                                        env=locName)

    def saveInterpolation(self, N, vectorStart, vectorEnd, pathOut):
        r"""
        Given two latent vactors, export the interpolated generations between
        them.

        Args:

            N (int): number of interpolation to make
            vectorStart (torch.tensor): start latent vector
            vectorEnd (torch.tensor): end latent vector
            pathOut (string): path where the images sould be saved
        """

        sizeStep = 1.0 / (N - 1)
        pathOut = os.path.splitext(pathOut)[0]

        vectorStart = vectorStart.view(1, -1, 1, 1)
        vectorEnd = vectorEnd.view(1, -1, 1, 1)

        nZeros = int(math.log10(N) + 1)

        for i in range(N):
            path = pathOut + str(i).zfill(nZeros) + ".jpg"
            t = i * sizeStep
            vector = (1 - t) * vectorStart + t * vectorEnd

            outImg = self.model.test(vector, getAvG=True, toCPU=True)
            self.visualizer.saveTensor(
                outImg, (outImg.size(2), outImg.size(3)), path)

    def visualizeNN(self,
                    N,
                    k,
                    featureExtractor,
                    imgTransform,
                    nnSearch,
                    names,
                    pathDB):
        r"""
        Visualize the nearest neighbors of some random generations

        Args:

            N (int): number of generation to make
            k (int): number of neighbors to fetch
            featureExtractor (nn.Module): feature extractor
            imgTransform (nn.Module): image transform module
            nnSearch (np.KDTree): serach tree for the features
            names (list): a match between an image index and its name
        """

        batchSize = 16
        nImages = 0

        vectorOut = []

        size = self.model.getSize()[0]

        transform = Transforms.Compose([NumpyResize((size, size)),
                                        NumpyToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))])

        dataset = None

        if os.path.splitext(pathDB)[1] == ".h5":
            dataset = H5Dataset(pathDB,
                                transform=Transforms.Compose(
                                [NumpyToTensor(),
                                 Transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))]))

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
        r"""
        Compute the nearest neighbors metric

        Args:

            N (int): number of generation to sample
            k (int): number of nearest neighbors to fetch
            featureExtractor (nn.Module): feature extractor
            imgTransform (nn.Module): image transform module
            nnSearch (np.KDTree): serach tree for the features
        """

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
