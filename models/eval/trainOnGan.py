import os
import json
import sys

import random

import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms
import torchvision.models as models

from ..gan_visualizer import GANVisualizer
from ..progressive_gan import ProgressiveGAN
from ..networks.constant_net import FeatureTransform
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, \
                          getNameAndPackage, prepareClassifier, \
                          printProgressBar, saveScore, parse_state_name

def updateParser(parser):

    parser.add_argument("-C", "--categoryName", dest="categoryName",
                        type=str, help="Class to classify")
    parser.add_argument("-I", "--nImgs", dest="nImgs",
                        type=int, help="number of images")

    return parser

def generateImagesFomConstraints(model, labels, labelName):

    output = []
    for label in labels:
        constraints = {labelName : label}
        input = model.buildNoiseDataWithConstraints(1, constraints)
        output.append(input)

    output = torch.cat(output, dim=0)
    return model.test(output, getAvG = True).detach()

def train(model,
          generator,
          criterion,
          optimizer,
          scheduler,
          imageTransform,
          labelWeights,
          num_imgs=70000):

    device = torch.device("cuda:0")
    batchSize = 16
    nRuns = int(num_imgs / batchSize)
    runLog = 100
    epochSize = int(nRuns / 3)

    accuracy = 0
    lastAc = 0
    model.train()

    for run in range(nRuns):

        labels = random.choices(labelWeights["values"], weights = labelWeights["weights"], k = batchSize)
        images = generateImagesFomConstraints(generator, labels,labelWeights["name"])

        labelInt = torch.tensor([labelWeights["match"][label] for label in labels],
                                dtype= torch.long, device = device)

        images = imageTransform(images)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labelInt)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        accuracy+= float(torch.sum(preds == labelInt.data).item())

        if run % runLog == (runLog-1):
            accuracy /= runLog*batchSize
            print("Iter %d accuracy %f" %(run, accuracy))
            lastAc= accuracy
            accuracy = 0

        if run % epochSize == (epochSize - 1):
            scheduler.step()

    return model, lastAc

def test(parser, visualisation = None):

    # Parameters
    parser = updateParser(parser)
    kwargs = vars(parser.parse_args())

    refSize = 224

    name =  getVal(kwargs,"name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs,"module", None)
    if module is None:
        raise ValueError("You need to input a module")

    pathStats = getVal(kwargs, "statsFile", None)
    if pathStats is None:
        raise ValueError("You need to input a pathStats")

    categoryName = getVal(kwargs, "categoryName", None)
    if categoryName is None:
        raise ValueError("You need to input a categoryName")

    scale              = getVal(kwargs, "scale", None)
    iter               = getVal(kwargs, "iter", None)
    checkPointDir      = getVal(kwargs, "dir", os.path.join('testNets', name))
    nImgs              = getVal(kwargs, "nImgs", 70000)
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
    generator = modelType(useGPU = True,
                      storeAVG = True,
                      **configData)

    generator.load(pathModel)

    with open(pathStats, 'rb') as file:
        statsDict = json.load(file)
        statsDict = statsDict[categoryName]

    nFeatures = len(statsDict)
    labelWeights = {"values":[], "weights":[], "match":{}, "name":categoryName}

    for label, value in statsDict.items():
        labelWeights["match"][label] = len(labelWeights["weights"])
        labelWeights["weights"].append(value)
        labelWeights["values"].append(label)

    print(labelWeights)
    print(statsDict)

    model = models.resnet34(pretrained=True)
    strModel = "resnet34"
    inFeatures = model.fc.in_features
    model.fc = torch.nn.Linear(inFeatures, nFeatures)
    model = torch.nn.DataParallel(model).to(torch.device("cuda:0"))

    # Criterion
    criterion = torch.nn.CrossEntropyLoss().to(torch.device("cuda:0"))

    #optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                betas = [0.9, 0.99], lr = 1e-2)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1)

    mean = [2 * x - 1 for x in [0.485, 0.456, 0.406]]
    std = [ 2 * x for x in [0.229, 0.224, 0.225]]
    upsamplingModule = torch.nn.DataParallel(FeatureTransform(mean, std, size = refSize)).to(torch.device("cuda:0"))

    model, last_ac = train(model, generator, criterion, optimizer, scheduler, upsamplingModule, labelWeights, num_imgs=nImgs)

    output_path = modelConfig[:-18] + "_ganTrained_" + categoryName + ".pt"
    outputDict = {"modelType": strModel,
                  "state_dict" : model.module.state_dict(),
                  "outFeatures": nFeatures,
                  "size": refSize,
                  "accuracy": last_ac,
                  "labels": labelWeights}

    torch.save(outputDict, output_path)
