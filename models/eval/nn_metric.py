# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ..networks.constant_net import FeatureTransform
from ..metrics.nn_score import buildFeatureExtractor
from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
    parse_state_name, getNameAndPackage, toStrKey, saveScore
from ..gan_visualizer import GANVisualizer
import torch.nn as nn
import torch
import os
import json

import pickle

import sys
sys.path.append("..")


def getModelName(pathConfig):

    pathConfig = os.path.basename(pathConfig)

    if pathConfig[-18:] != '_train_config.json':
        raise ValueError("Invalid configuration name")

    return pathConfig[:-18]


def update_parser(parser):

    parser.add_argument('--showNN', action='store_true')
    parser.add_argument('--size', help="Image size",
                        type=int, dest="size", default=224)
    parser.add_argument('-f', '--featureExtractor', help="Path to the feature \
                        extractor",
                        type=str, dest="featureExtractor")


def test(parser, visualisation=None):

    update_parser(parser)

    kwargs = vars(parser.parse_args())
    # Parameters
    name = getVal(kwargs, "name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    trainingConfig = getVal(kwargs, "config", None)
    if trainingConfig is None:
        raise ValueError("You need to input a configuration file")

    pathNNFeatureExtractor = getVal(kwargs, "featureExtractor", None)
    if pathNNFeatureExtractor is None:
        raise ValueError("You need to give a feature extractor")

    # Mandatory fields
    checkPointDir = os.path.join(kwargs["dir"], name)
    scale = getVal(kwargs, "scale", None)
    iter = getVal(kwargs, "iter", None)

    checkpointData = getLastCheckPoint(
        checkPointDir, name, scale=scale, iter=iter)

    if checkpointData is None:
        if scale is not None or iter is not None:
            raise FileNotFoundError("Not checkpoint found for model " + name
                                    + " at directory " + dir + " for scale " +
                                    str(scale) + " at iteration " + str(iter))
        raise FileNotFoundError(
            "Not checkpoint found for model " + name + " at directory " + dir)

    modelConfig, pathModel, _ = checkpointData

    if scale is None or iter is None:
        _, scale, iter = parse_state_name(pathModel)

    # Feature extraction

    # Look for NN data

    with open(trainingConfig, 'rb') as file:
        wholeConfig = json.load(file)

    pathDB = wholeConfig.get("pathDB", None)
    if pathDB is None:
        raise ValueError("No training database found")

    partitionValue = wholeConfig.get("partitionValue", None)
    partitionValue = getVal(kwargs, "partition_value", None)

    pathOutFeatures = os.path.splitext(pathNNFeatureExtractor)[0] + "_" + \
        os.path.splitext(os.path.basename(pathDB))[0] + "_" + \
        str(kwargs['size']) + \
        toStrKey(partitionValue) + "_features.pkl"

    if not os.path.isfile(pathNNFeatureExtractor) \
            or not os.path.isfile(pathOutFeatures):
        raise FileNotFoundError("No model found at " + pathOutFeatures)

    print("Loading model " + pathModel)
    modelPackage, modelName = getNameAndPackage(module)
    modelType = loadmodule(modelPackage, modelName)
    visualizer = GANVisualizer(
        pathModel, modelConfig, modelType, visualisation)

    print("NN model found !  " + pathNNFeatureExtractor)
    featureExtractor, mean, std = buildFeatureExtractor(pathNNFeatureExtractor)

    imgTransform = nn.DataParallel(FeatureTransform(
        mean, std, kwargs['size'])).to(torch.device("cuda:0"))
    featureExtractor = nn.DataParallel(
        featureExtractor).to(torch.device("cuda:0"))

    with open(pathOutFeatures, 'rb') as file:
        nnSearch, names = pickle.load(file)

    if kwargs['showNN']:
        print("Retriving 10 neighbors for visualization")
        visualizer.visualizeNN(10, 5, featureExtractor,
                               imgTransform, nnSearch, names, pathDB)
        print("Ready, please check out visdom main environement")
    else:

        outMetric = visualizer.exportNN(
            1600, 8, featureExtractor, imgTransform, nnSearch)
        outPath = modelConfig[:-18] + "_nn_metric.json"

        saveScore(outPath, list(outMetric), scale,
                  iter, pathNNFeatureExtractor)
