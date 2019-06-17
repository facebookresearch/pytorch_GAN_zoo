# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import torchvision

from ..metrics.inception_score import InceptionScore
from ..utils.utils import printProgressBar
from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
    parse_state_name, getNameAndPackage, saveScore
from ..networks.constant_net import FeatureTransform


def test(parser, visualisation=None):

    kwargs = vars(parser.parse_args())

    # Are all parameters available ?
    name = getVal(kwargs, "name", None)
    if name is None and not kwargs['selfNoise']:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    # Loading the model
    scale = getVal(kwargs, "scale", None)

    if name is not None:
        iter = getVal(kwargs, "iter", None)

        checkPointDir = os.path.join(kwargs["dir"], name)
        checkpointData = getLastCheckPoint(
            checkPointDir, name, scale=scale, iter=iter)

        if checkpointData is None:
            print(scale, iter)
            if scale is not None or iter is not None:
                raise FileNotFoundError("Not checkpoint found for model "
                                        + name + " at directory " + dir +
                                        " for scale " + str(scale) +
                                        " at iteration " + str(iter))
            raise FileNotFoundError(
                "Not checkpoint found for model " + name + " at directory "
                + dir)

        modelConfig, pathModel, _ = checkpointData
        with open(modelConfig, 'rb') as file:
            configData = json.load(file)

        modelPackage, modelName = getNameAndPackage(module)
        modelType = loadmodule(modelPackage, modelName)

        model = modelType(useGPU=True,
                          storeAVG=True,
                          **configData)

        if scale is None or iter is None:
            _, scale, iter = parse_state_name(pathModel)

        print("Checkpoint found at scale %d, iter %d" % (scale, iter))
        model.load(pathModel)

    elif scale is None:
        raise AttributeError("Please provide a scale to compute the noise of \
        the dataset")

    # Building the score instance
    classifier = torchvision.models.inception_v3(pretrained=True).cuda()
    scoreMaker = InceptionScore(classifier)

    batchSize = 16
    nBatch = 1000

    refMean = [2*p - 1 for p in[0.485, 0.456, 0.406]]
    refSTD = [2*p for p in [0.229, 0.224, 0.225]]
    imgTransform = FeatureTransform(mean=refMean,
                                    std=refSTD,
                                    size=299).cuda()

    print("Computing the inception score...")
    for index in range(nBatch):

        inputFake = model.test(model.buildNoiseData(batchSize)[0],
                               toCPU=False, getAvG=True)

        scoreMaker.updateWithMiniBatch(imgTransform(inputFake))
        printProgressBar(index, nBatch)

    printProgressBar(nBatch, nBatch)
    print("Merging the results, please wait it can take some time...")
    score = scoreMaker.getScore()

    # Now printing the results
    print(score)

    # Saving the results
    if name is not None:

        outPath = os.path.join(checkPointDir, name + "_swd.json")
        saveScore(outPath, score,
                  scale, iter)
