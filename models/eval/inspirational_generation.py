import os
import torch
import torchvision
import json

from ..gan_visualizer import GANVisualizer
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, getNameAndPackage
from ..metrics.nn_score import buildFeatureExtractor
from ..networks.constant_net import FeatureTransform

from PIL import Image
import torchvision
import torchvision.transforms as Transforms
import torchvision.models as models
from torch.utils.serialization import load_lua

import torch.nn as nn
import torch.nn.functional as F


def pil_loader(path):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IDModule(nn.Module):

    def __init__(self):

        super(IDModule, self).__init__()
        #mself.dummy = nn.Conv2d(1,1,1,1)

    def forward(self, x):
        return x


def updateParser(parser):

    parser.add_argument('-f', '--featureExtractor', help="Partition's value", nargs='*',
                        type=str, dest="featureExtractor")
    parser.add_argument('--inputImage', type=str, dest="inputImage",
                        help="Path to the input image.")
    parser.add_argument('-N', type=int, dest="nRuns",
                        help="Number of gradient descent to run",
                        default=1)
    parser.add_argument('-l', type=float, dest="learningRate",
                        help="Learning rate",
                        default=1)
    parser.add_argument('-S', '--suffix', type=str, dest='suffix',
                        help="Output's suffix", default="inspiration")
    parser.add_argument('-R', '--rLoss', type=float, dest='lambdaD',
                        help="Realism penalty", default=0.03)
    parser.add_argument('--nSteps', type=int, dest='nSteps',
                        help="Number of steps", default=6000)
    parser.add_argument('--weights', type=float, dest='weights',
                        nargs='*', help="Weight of each classifier. Default value is one.\
                        If specified, the number of weights must match.")
    parser.add_argument('--random_search', help='Random search',
                        action='store_true')
    parser.add_argument('--save_descent', help='Save descent',
                        action='store_true')

    return parser

def gradientDescentOnInput(model,
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

    noiseOut = model.test(varNoise, getAvG=True, toCPU=False)

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
                featureExtractors[i]).train().to(model.device)

        imageTransforms[i] = nn.DataParallel(
            imageTransforms[i]).to(model.device)

        featuresIn.append(featureExtractors[i](
            imageTransforms[i](input.to(model.device))).detach())

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

        noiseOut = model.avgG(varNoise)
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


def test(parser, visualisation=None):

    parser = updateParser(parser)

    kwargs = vars(parser.parse_args())

    # Parameters
    name = getVal(kwargs, "name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    imgPath = getVal(kwargs, "inputImage", None)
    if imgPath is None:
        raise ValueError("You need to input an image path")

    scale = getVal(kwargs, "scale", None)
    iter = getVal(kwargs, "iter", None)
    nRuns = getVal(kwargs, "nRuns", 1)

    checkPointDir = os.path.join(kwargs["dir"], name)
    checkpointData = getLastCheckPoint(checkPointDir,
                                       name,
                                       scale=scale,
                                       iter=iter)
    weights = getVal(kwargs, 'weights', None)

    if checkpointData is None:
        raise FileNotFoundError(
            "Not checkpoint found for model " + name + " at directory " + dir)

    modelConfig, pathModel, _ = checkpointData

    keysLabels = None
    with open(modelConfig, 'rb') as file:
        keysLabels = json.load(file)["attribKeysOrder"]
    if keysLabels is None:
        keysLabels = {}

    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)

    visualizer = GANVisualizer(
        pathModel, modelConfig, modelType, visualisation)

    # Load the image
    targetSize = visualizer.model.getSize()

    baseTransform = Transforms.Compose([Transforms.Resize((targetSize, targetSize)),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inputs = [imgPath]

    for path in inputs:

        img = pil_loader(path)
        input = baseTransform(img)
        input = input.view(1, input.size(0), input.size(1), input.size(2))

    pathsModel = getVal(kwargs, "featureExtractor", None)
    featureExtractors = []
    imgTransforms = []

    if weights is not None:
        if pathsModel is None or len(pathsModel) != len(weights):
            raise ArgumentError(
                "The number of weights must match the number of models")

    if pathsModel is not None:
        for path in pathsModel:
            if path == "id":
                featureExtractor = IDModule()
                imgTransform = IDModule()
            else:
                featureExtractor, mean, std = buildFeatureExtractor(
                    path, resetGrad=True)
                imgTransform = FeatureTransform(mean, std, size=128)  # None)
            featureExtractors.append(featureExtractor)
            imgTransforms.append(imgTransform)
    else:
        featureExtractors = IDModule()
        imgTransforms = IDModule()

    outVectorList = []
    basePath = os.path.splitext(imgPath)[0] + "_" + kwargs['suffix']

    if not os.path.isdir(basePath):
        os.mkdir(basePath)

    basePath = os.path.join(basePath, os.path.basename(basePath))

    outDictData = {}
    outPathDescent = None

    for i in range(nRuns):

        if kwargs['save_descent']:
            outPathDescent = os.path.join(
                os.path.dirname(basePath), "descent_" + str(i))
            if not os.path.isdir(outPathDescent):
                os.mkdir(outPathDescent)

        img, vector, loss = visualizer.model.gradientDescentOnInput(input,
                                                                    featureExtractors,
                                                                    imgTransforms,
                                                                    visualizer=visualisation,
                                                                    lambdaD=kwargs['lambdaD'],
                                                                    nSteps=kwargs['nSteps'],
                                                                    weights=weights,
                                                                    randomSearch=kwargs['random_search'],
                                                                    lr=kwargs['learningRate'],
                                                                    outPathSave=outPathDescent)
        outVectorList.append(vector)
        path = basePath + "_" + str(i) + ".jpg"
        visualisation.saveTensor(img, (img.size(2), img.size(3)), path)
        outDictData[os.path.splitext(os.path.basename(path))[0]] = loss

    outVectors = torch.cat(outVectorList, dim=0)
    outVectors = outVectors.view(outVectors.size(0), -1)
    outVectors *= torch.rsqrt((outVectors**2).mean(dim=1, keepdim=True))

    barycenter = outVectors.mean(dim=0)
    barycenter *= torch.rsqrt((barycenter**2).mean())
    meanAngles = (outVectors * barycenter).mean(dim=1)
    meanDist = torch.sqrt(((barycenter-outVectors)**2).mean(dim=1)).mean(dim=0)
    outDictData["Barycenter"] = {"meanDist": meanDist.item(
    ), "stdAngles": meanAngles.std().item(), "meanAngles": meanAngles.mean().item()}

    path = basePath + "_data.json"
    outDictData["kwargs"] = kwargs

    with open(path, 'w') as file:
        json.dump(outDictData, file, indent=2)

    pathVectors = basePath + "vectors.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))
