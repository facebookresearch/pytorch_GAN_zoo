import os
import json
import sys

import torch
import torch.nn.functional as F

from ..gan_visualizer import GANVisualizer
from ..progressive_gan import ProgressiveGAN
from ..networks.constant_net import FeatureTransform
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, \
                          getNameAndPackage, prepareClassifier, \
                          printProgressBar, saveScore, parse_state_name

def generateImagesFomConstraints(model, nImages, constraints):

    input = model.buildNoiseDataWithConstraints(nImages, constraints)
    return model.test(input, getAvG = True).detach()

def getModelName(pathConfig):

    pathConfig = os.path.basename(pathConfig)

    if pathConfig[-18:] != '_train_config.json':
        raise ValueError("Invalid configuration name")

    return pathConfig[:-18]

def test(parser, visualisation = None):

    # Parameters
    parser.add_argument('-f','--featureExtractor', help="Partition's value",
                        type=str, dest="featureExtractor")
    kwargs = vars(parser.parse_args())

    name =  getVal(kwargs,"name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs,"module", None)
    if module is None:
        raise ValueError("You need to input a module")

    pathClassifier= getVal(kwargs, "featureExtractor", None)
    if pathClassifier is None:
        raise ValueError("You need to give a feature extractor")

    pathStats          = getVal(kwargs, "statsFile", None)
    scale              = getVal(kwargs, "scale", None)
    iter               = getVal(kwargs, "iter", None)
    checkPointDir      = os.path.join(kwargs["dir"], modelLabel)
    checkpointData     = getLastCheckPoint(checkPointDir,
                                           name,
                                           scale = scale,
                                           iter = iter)


    if checkpointData is None:
        raise FileNotFoundError("Not checkpoint found for model " + name + " at directory " + dir)

    modelConfig, pathModel, _ = checkpointData

    with open(modelConfig, 'rb') as file:
        configData = json.load(file)

    if scale is None or iter is None:
        _, scale, iter = parse_state_name(pathModel)

    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)
    model = modelType(useGPU = True,
                      storeAVG = True,
                      **configData)

    model.load(pathModel)

    modelState = torch.load(pathClassifier)
    classifierType = loadmodule('torchvision.models', modelState["modelType"], prefix ='')
    outFeatures = modelState["outFeatures"]
    refSize = modelState["size"]

    classifier = prepareClassifier(classifierType, outFeatures)
    classifier.load_state_dict(modelState["state_dict"])
    classifier = torch.nn.DataParallel(classifier).to(torch.device("cuda:0"))

    mean = [2 * x - 1 for x in [0.485, 0.456, 0.406]]
    std = [ 2 * x for x in [0.229, 0.224, 0.225]]
    upsamplingModule = torch.nn.DataParallel(FeatureTransform(mean, std, size = refSize)).to(torch.device("cuda:0"))

    classifierDict = modelState["labels"]
    categoryName = list(classifierDict.keys())[0]

    labels = classifierDict[categoryName]["values"]

    print(modelState.keys())

    nRuns = 16000
    batchSize = 16

    nLabels = len(labels)
    if pathStats is None:
        iterStep = int(nRuns / nLabels)
        stepsRun = [i * iterStep for i in range(nLabels)]
    else:

        with open(pathStats, 'rb') as file:
            statsDict = json.load(file)
            statsDict = statsDict[categoryName]

        nData = sum(val for key, val in statsDict.items())
        stepsRun = [0]
        currentStep = 0
        for label in labels[:-1]:
            sizeLabel = statsDict[label]
            totSize = int((nRuns * sizeLabel) / nData)
            currentStep+= totSize
            stepsRun.append(currentStep)

    nImages = 0
    currIndexLabel = 0
    toValids = 0

    outResults = {}

    stepsRun.append(nRuns)
    print(labels, stepsRun)

    confusion = torch.zeros(nLabels, nLabels)
    confusionDict ={}

    for currIndexLabel in range(nLabels):

        labelName = labels[currIndexLabel]

        nextStep = stepsRun[currIndexLabel + 1]

        constraints = {categoryName : labelName}
        sumProba = 0
        sumValid = 0

        start = nImages

        while nImages < nextStep:

            printProgressBar(nImages, nRuns)

            images = generateImagesFomConstraints(model, batchSize, constraints)
            probabilities = F.softmax(classifier(upsamplingModule(images)).detach(), dim=1)
            preds = torch.argmax(probabilities, dim=1)

            sumProba += probabilities[:,currIndexLabel].sum().item()
            sumValid += torch.sum(preds == currIndexLabel).item()

            for p in range(nLabels):
                confusion[currIndexLabel, p] += torch.sum(preds == p).item()

            nImages += batchSize

        delta = float(nImages - start)
        meanProba = sumProba / delta
        accuracy = sumValid /delta

        confusion[currIndexLabel] /= delta

        confusionDict[labelName] = {labels[k] : confusion[currIndexLabel, k].item() for k in range(nLabels) }#if labels[k] not in ["b'id_gridfs_5'", "b'id_gridfs_6'"]}

        toValids += sumValid
        outResults[labelName] = {"accuracy": accuracy, "meanProba": meanProba}


    printProgressBar(nRuns, nRuns)

    outResults["globalAccuracy"] = toValids / nImages
    outResults["confusion"] = confusionDict
    outResults["Training accuracy"] = modelState["accuracy"].item()

    outPath = modelConfig[:-18] + "_conformity_metric.json"
    print(outResults)
    saveScore(outPath, outResults, scale, iter, pathClassifier)
