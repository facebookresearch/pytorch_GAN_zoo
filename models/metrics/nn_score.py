# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import random
import pickle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as Transforms

import scipy
import scipy.spatial
import numpy

from models.datasets.attrib_dataset import AttribDataset
from ..datasets.hd5 import H5Dataset
from ..datasets.attrib_dataset import AttribDataset
from ..loss_criterions.ac_criterion import ACGANCriterion
from ..utils.utils import printProgressBar, loadmodule

random.seed()


def getStatsOnDataset(attributes):

    stats = {}

    for name, data in attributes.items():
        for key, value in data.items():

            if key not in stats:
                stats[key] = {}

            if value not in stats[key]:
                stats[key][value] = 0

            stats[key][value] += 1
    return stats


def updateStatsWithData(stats, item):

    for key, value in item.items():
        stats[key][value] += 1


def buildTrainValTest(pathAttrib,
                      shareTrain=0.8,
                      shareVal=0.2):

    with open(pathAttrib, 'rb') as file:
        data = json.load(file)

    stats = getStatsOnDataset(data)

    shareTest = max(0., 1. - shareTrain - shareVal)

    targetTrain = {key: {value: stats[key][value] * shareTrain
                         for value in stats[key]} for key in stats}
    targetVal = {key: {value: stats[key][value] * shareVal
                       for value in stats[key]} for key in stats}
    targetTest = {key: {value: stats[key][value] * shareTest
                        for value in stats[key]} for key in stats}

    keys = [key for key in data.keys()]
    random.shuffle(keys)

    outTrain = {}
    outVal = {}
    outTest = {}

    trainStats = {key: {value: 0 for value in stats[key]} for key in stats}
    valStats = {key: {value: 0 for value in stats[key]} for key in stats}
    testStats = {key: {value: 0 for value in stats[key]} for key in stats}

    for name in keys:

        scoreTrain = 0
        scoreVal = 0
        scoreTest = 0

        for category in data[name]:
            label = data[name][category]
            deltaTrain = max(0, targetTrain[category][label] - trainStats[category][label]) / (
                targetTrain[category][label] + 1e-8)
            deltaVal = max(0, targetVal[category][label] - valStats[category]
                           [label]) / (targetVal[category][label] + 1e-8)
            deltaTest = max(0, targetTest[category][label] - testStats[category][label]) / (
                targetTest[category][label] + 1e-8)

            scoreTrain += deltaTrain**2
            scoreVal += deltaVal**2
            scoreTest += deltaTest**2

        if scoreTrain >= 0.999 or scoreTrain >= max(scoreVal, scoreTest):
            outTrain[name] = data[name]
            updateStatsWithData(trainStats, data[name])
        elif scoreVal >= scoreTest:
            outVal[name] = data[name]
            updateStatsWithData(valStats, data[name])
        else:
            outTest[name] = data[name]
            updateStatsWithData(testStats, data[name])

    stats = {"Train": trainStats, "Val": valStats, "Test": testStats}

    return outTrain, outVal, outTest, stats


def buildFeatureMaker(pathDB,
                      pathTrainAttrib,
                      pathValAttrib,
                      specificAttrib=None,
                      visualisation=None):

    # Parameters
    batchSize = 16
    nEpochs = 3
    learningRate = 1e-4
    beta1 = 0.9
    beta2 = 0.99
    device = torch.device("cuda:0")
    n_devices = torch.cuda.device_count()

    # Model
    resnet18 = models.resnet18(pretrained=True)
    resnet18.train()

    # Dataset
    size = 224
    transformList = [Transforms.Resize((size, size)),
                     Transforms.RandomHorizontalFlip(),
                     Transforms.ToTensor(),
                     Transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    transform = Transforms.Compose(transformList)

    dataset = AttribDataset(pathDB, transform=transform,
                            attribDictPath=pathTrainAttrib,
                            specificAttrib=specificAttrib,
                            mimicImageFolder=False)

    validationDataset = AttribDataset(pathDB, transform=transform,
                                      attribDictPath=pathValAttrib,
                                      specificAttrib=specificAttrib,
                                      mimicImageFolder=False)

    print("%d training images detected, %d validation images detected"
          % (len(dataset), len(validationDataset)))

    # Optimization
    optimizer = torch.optim.Adam(resnet18.parameters(),
                                 betas=[beta1, beta2],
                                 lr=learningRate)

    lossMode = ACGANCriterion(dataset.getKeyOrders())

    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, lossMode.getInputDim())
    resnet18 = nn.DataParallel(resnet18).to(device)

    # Visualization data
    lossTrain = []
    lossVal = []
    iterList = []
    tokenTrain = None
    tokenVal = None
    step = 0
    tmpLoss = 0

    for epoch in range(nEpochs):

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True,
                                             num_workers=n_devices)

        for iter, data in enumerate(loader):

            optimizer.zero_grad()

            inputs_real, labels = data
            inputs_real = inputs_real.to(device)
            labels = labels.to(device)

            predictedLabels = resnet18(inputs_real)

            loss = lossMode.getLoss(predictedLabels, labels)

            tmpLoss += loss.item()

            loss.backward()
            optimizer.step()

            if step % 100 == 0 and visualisation is not None:

                divisor = 100
                if step == 0:
                    divisor = 1
                lossTrain.append(tmpLoss / divisor)
                iterList.append(step)
                tokenTrain = visualisation.publishLinePlot([('lossTrain', lossTrain)], iterList,
                                                           name="Loss train",
                                                           window_token=tokenTrain,
                                                           env="main")

                validationLoader = torch.utils.data.DataLoader(validationDataset,
                                                               batch_size=batchSize,
                                                               shuffle=True,
                                                               num_workers=n_devices)

                resnet18.eval()
                lossEval = 0
                i = 0
                for valData in validationLoader:

                    inputs_real, labels = data
                    inputs_real = inputs_real.to(device)
                    labels = labels.to(device)
                    lossEval += lossMode.getLoss(predictedLabels,
                                                 labels).item()
                    i += 1

                    if i == 100:
                        break

                lossEval /= i
                lossVal.append(lossEval)
                tokenVal = visualisation.publishLinePlot([('lossValidation', lossVal)], iterList,
                                                         name="Loss validation",
                                                         window_token=tokenVal,
                                                         env="main")
                resnet18.train()

                print("[%5d ; %5d ] Loss train : %f ; Loss validation %f"
                      % (epoch, iter, tmpLoss / divisor, lossEval))
                tmpLoss = 0

            step += 1

    return resnet18.module


def cutModelHead(model):

    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)

    return model


def buildFeatureExtractor(pathModel, resetGrad=True):

    modelData = torch.load(pathModel)

    fullDump = modelData.get("fullDump", False)
    if fullDump:
        model = modelData['model']
    else:
        modelType = loadmodule(
            modelData['package'], modelData['network'], prefix='')
        model = modelType(**modelData['kwargs'])
        model = cutModelHead(model)
        model.load_state_dict(modelData['data'])

    for param in model.parameters():
        param.requires_grad = resetGrad

    mean = modelData['mean']
    std = modelData['std']

    return model, mean, std


def saveFeatures(model,
                 imgTransform,
                 pathDB,
                 pathMask,
                 pathAttrib,
                 outputFile,
                 pathPartition=None,
                 partitionValue=None):

    batchSize = 16

    transformList = [Transforms.ToTensor(),
                     Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transformList)

    device = torch.device("cuda:0")
    n_devices = torch.cuda.device_count()

    parallelModel = nn.DataParallel(model).to(device).eval()
    parallelTransorm = nn.DataParallel(imgTransform).to(device)

    if os.path.splitext(pathDB)[1] == ".h5":
        dataset = H5Dataset(pathDB,
                            transform=transform,
                            pathDBMask=pathMask,
                            partition_path=pathPartition,
                            partition_value=partitionValue)

    else:
        dataset = AttribDataset(pathDB, transform=transform,
                                attribDictPath=pathAttrib,
                                specificAttrib=None,
                                mimicImageFolder=False,
                                pathMask=pathMask)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batchSize,
                                         shuffle=False,
                                         num_workers=n_devices)

    outFeatures = []

    nImg = 0
    totImg = len(dataset)

    for item in loader:

        if len(item) == 3:
            data, label, mask = item
        else:
            data, label = item
            mask = None

        printProgressBar(nImg, totImg)
        features = parallelModel(parallelTransorm(
            data)).detach().view(data.size(0), -1).cpu()
        outFeatures.append(features)

        nImg += batchSize

    printProgressBar(totImg, totImg)

    import sys
    sys.setrecursionlimit(10000)

    outFeatures = torch.cat(outFeatures, dim=0).numpy()
    tree = scipy.spatial.KDTree(outFeatures, leafsize=10)
    names = [dataset.getName(x) for x in range(totImg)]
    with open(outputFile, 'wb') as file:
        pickle.dump([tree, names], file)
