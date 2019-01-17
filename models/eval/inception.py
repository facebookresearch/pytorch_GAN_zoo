import os
import json

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as Transforms

from ..metrics.inception_score import InceptionScore
from ..utils.utils import printProgressBar, getVal, loadmodule, \
                          getLastCheckPoint, parse_state_name,  \
                          getNameAndPackage, saveScore, prepareClassifier

from ..networks.constant_net import FeatureTransform
import sys

def test(parser, visualisation = None):

    miniBatchSize = 16
    parser.add_argument('-f','--featureExtractor', help="Partition's value",
                        type=str, dest="featureExtractor")

    kwargs = vars(parser.parse_args())
    # Parameters
    name =  getVal(kwargs,"name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs,"module", None)
    if module is None:
        raise ValueError("You need to input a module")

    pathClassifier= getVal(kwargs, "featureExtractor", None)
    if pathClassifier is None:
        raise ValueError("You need to give a feature extractor")

    # Load the classifier
    modelState = torch.load(pathClassifier)
    classifierType = loadmodule('torchvision.models', modelState["modelType"], prefix ='')
    outFeatures = modelState["outFeatures"]
    refSize = modelState["size"]

    nImg = outFeatures * 320
    nMiniBatches = int(nImg / miniBatchSize)

    classifier = prepareClassifier(classifierType, outFeatures)
    classifier.load_state_dict(modelState["state_dict"])
    classifier = torch.nn.DataParallel(classifier).to(torch.device("cuda:0"))

    #Mandatory fields
    checkPointDir      = getVal(kwargs, "dir", os.path.join('testNets', name))
    scale              = getVal(kwargs, "scale", None)
    iter               = getVal(kwargs, "iter", None)

    checkpointData     = getLastCheckPoint(checkPointDir, name, scale = scale, iter = iter)


    if checkpointData is None:
        print(scale, iter)
        if scale is not None or iter is not None:
            raise FileNotFoundError("Not checkpoint found for model " + name \
                                    + " at directory " + dir + " for scale " + \
                                    str(scale) + " at iteration " + str(iter))
        raise FileNotFoundError("Not checkpoint found for model " + name + " at directory " + dir)

    modelConfig, pathModel, _ = checkpointData

    if scale is None or iter is None:
        _, scale, iter = parse_state_name(pathModel)

    with open(modelConfig, 'rb') as file:
        configData = json.load(file)

    modelPackage, modelName = getNameAndPackage(module)
    modelType = loadmodule(modelPackage, modelName)
    model = modelType(useGPU = True,
                      storeAVG = True,
                      **configData)

    model.load(pathModel)

    InceptionMetric = InceptionScore(classifier)

    mean = [2 * x - 1 for x in [0.485, 0.456, 0.406]]
    std = [ 2 * x for x in [0.229, 0.224, 0.225]]
    upsamplingModule = nn.DataParallel(FeatureTransform(mean, std, size = refSize)).to(torch.device("cuda:0"))

    numWorkers = 2
    device = torch.device("cuda:0")
    n_devices = torch.cuda.device_count()
    classifier = nn.DataParallel(classifier)
    classifier.to(device)

    for i in range(nMiniBatches):

        inputFake = model.test(model.buildNoiseData(miniBatchSize)[0]).to(device)

        visualisation.publishTensors(inputFake.cpu(), (256, 256))
        inputFake = upsamplingModule(inputFake)
        InceptionMetric.updateWithMiniBatch(inputFake)

        printProgressBar(i, nMiniBatches)

    printProgressBar(nMiniBatches, nMiniBatches)

    score = InceptionMetric.getScore()
    outPath = modelConfig[:-18] + "_inception_metric.json"
    saveScore(outPath, [score], scale, iter, pathClassifier)
