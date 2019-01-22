import os
import json

import time

import torch
import torchvision.transforms as Transforms

from ..metrics.laplacian_swd import LaplacianSWDMetric
from ..utils.utils import printProgressBar
from ..datasets.attrib_dataset import AttribDataset
from ..datasets.hd5 import H5Dataset
from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
                          parse_state_name, getNameAndPackage
from ..utils.image_transform import NumpyResize


def test(parser, visualisation=None):

    parser.add_argument('--selfNoise', action='store_true',
                        help="Compute the inner noise of the dataset")
    kwargs = vars(parser.parse_args())

    # Are all parameters available ?
    name = getVal(kwargs, "name", None)
    if name is None and not kwargs['selfNoise']:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    trainingConfig = getVal(kwargs, "config", None)
    if trainingConfig is None:
        raise ValueError("You need to input a configuration file")

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
    depthPyramid = min(scale, 4)
    SWDMetric = LaplacianSWDMetric(7, 128, depthPyramid)

    # Building the dataset
    with open(trainingConfig, 'rb') as file:
        wholeConfig = json.load(file)

    pathPartition = wholeConfig.get("pathPartition", None)
    partitionValue = wholeConfig.get("partitionValue", None)

    partitionValue = getVal(kwargs, "partition_value", None)

    # Training dataset properties
    pathDB = wholeConfig["pathDB"]
    attribDict = wholeConfig.get('pathAttrib', None)
    size = 2**(2 + scale)
    db_transform = Transforms.Compose([NumpyResize((size, size)),
                                       Transforms.ToTensor(),
                                       Transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])

    if os.path.splitext(pathDB)[1] == '.h5':
        dataset = H5Dataset(pathDB,
                            transform=db_transform,
                            partition_path=pathPartition,
                            partition_value=partitionValue)
    else:
        dataset = AttribDataset(pathdb=pathDB,
                                transform=db_transform,
                                attribDictPath=attribDict)

    # Metric parameters
    nImagesSampled = min(len(dataset), 16000)
    batchSize = 16
    maxBatch = nImagesSampled / batchSize

    dbLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                           num_workers=2, shuffle=True)

    if kwargs['selfNoise']:

        print("Computing the inner noise of the dataset...")
        loader2 = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                              num_workers=2, shuffle=True)

        for item, data in enumerate(zip(dbLoader, loader2)):

            if item > maxBatch:
                break

            real, fake = data
            SWDMetric.updateWithMiniBatch(real[0], fake[0])
            printProgressBar(item, maxBatch)

    else:

        print("Generating the fake dataset...")
        for item, data in enumerate(dbLoader, 0):

            if item > maxBatch:
                break

            inputsReal, _ = data
            inputFake = model.test(model.buildNoiseData(
                inputsReal.size(0))[0], toCPU=False, getAvG=True)

            SWDMetric.updateWithMiniBatch(inputFake, inputsReal)
            printProgressBar(item, maxBatch)

    printProgressBar(maxBatch, maxBatch)
    print("Merging the results, please wait it can take some time...")
    score = SWDMetric.getScore()

    # Saving the results
    outPath = os.path.join(checkPointDir, name + "_swd.json")
    flagPath = outPath + ".flag"

    while os.path.isfile(flagPath):
        time.sleep(1)

    open(flagPath, 'a').close()
    if os.path.isfile(outPath):
        with open(outPath, 'rb') as file:
            outResults = json.load(file)

        if not isinstance(outResults, dict):
            outResults = {}
    else:
        outResults = {}

    if str(scale) not in outResults:
        print("coin")
        outResults[str(scale)] = {}

    outResults[str(scale)][str(iter)] = score

    with open(outPath, 'w') as file:
        json.dump(outResults, file, indent=2)

    os.remove(flagPath)

    print("")
    print("score (from highest res to lowest): " + str(score))
    print("")
