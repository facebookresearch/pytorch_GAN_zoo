import os
import json
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms

from ..datasets.hd5 import H5Dataset
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, \
                          getNameAndPackage, prepareClassifier, \
                          printProgressBar, saveScore, parse_state_name
from utils.image_transform import NumpyResize

def getModelName(pathConfig):

    pathConfig = os.path.basename(pathConfig)

    if pathConfig[-18:] != '_train_config.json':
        raise ValueError("Invalid configuration name")

    return pathConfig[:-18]

def updateParser(parser):

    parser.add_argument('-f','--featureExtractor', help="Partition's value",
                        type=str, dest="featureExtractor")
    parser.add_argument("-C", "--categoryName", dest="categoryName",
                        type=str, help="Class to classify")

    return parser

def buildLabelConverted(keyOrdersSource, targetMatch):

    return [targetMatch[x] for x in keyOrdersSource]


def test(parser, visualisation = None):

    # Parameters
    parser = updateParser(parser)
    kwargs = vars(parser.parse_args())

    name =  getVal(kwargs,"name", None)
    if name is None:
        raise ValueError("You need to input a name")

    trainingConfig = getVal(kwargs,"config", None)
    if trainingConfig is None:
        raise ValueError("You need to input a configuration file")

    with open(trainingConfig, 'rb') as file:
        wholeConfig = json.load(file)

    pathDB = wholeConfig.get("pathDB", None)
    if pathDB is None:
        raise ValueError("No training database found")

    categoryName = getVal(kwargs, "categoryName", None)
    if categoryName is None:
        raise ValueError("You need to input a categoryName")

    pathPartition  = wholeConfig.get("pathPartition", None)
    partitionValue = wholeConfig.get("partitionValue", None)
    pathAttrib     = wholeConfig.get("pathAttrib", None)

    partitionValue = getVal(kwargs, "partition_value", None)
    pathAttrib     = getVal(kwargs, "statsFile", pathAttrib)
    pathVal        = getVal(kwargs, "valDatasetPath", None)
    pathPartVal    = getVal(kwargs, "valPartitionPath", None)
    checkPointDir  = getVal(kwargs, "dir", os.path.join('testNets', name))

    specificAttrib = [categoryName]

    # Dataset
    refSize = 224
    db_transform = Transforms.Compose([NumpyResize(refSize),
                                       Transforms.ToTensor(),
                                       Transforms.Normalize(mean =(0.485, 0.456, 0.406),
                                                            std  = (0.229, 0.224, 0.225))
                                       ])

    if os.path.splitext(pathDB)[1] == '.h5':
        dataset = H5Dataset(pathDB,
                            transform = db_transform,
                            partition_path = pathPartition,
                            partition_value = partitionValue,
                            stats_file = pathAttrib,
                            specificAttrib = specificAttrib)
    else:
        dataset = AttribDataset(pathdb=pathDB,
                                transform = db_transform,
                                attribDictPath = pathAttrib,
                                specificAttrib = specificAttrib)

    dbLoader = torch.utils.data.DataLoader(dataset,
                                batch_size= 16,
                                shuffle=True,
                                num_workers= torch.cuda.device_count())

    pathClassifier = os.path.join(checkPointDir, name) + "_ganTrained_" + categoryName + ".pt"

    modelState = torch.load(pathClassifier)
    classifierType = loadmodule('torchvision.models', modelState["modelType"], prefix ='')
    outFeatures = modelState["outFeatures"]
    refSize = modelState["size"]

    classifier = prepareClassifier(classifierType, outFeatures)
    classifier.load_state_dict(modelState["state_dict"])
    classifier = torch.nn.DataParallel(classifier).to(torch.device("cuda:0"))

    classifierDict = modelState["labels"]
    classifierlabels = classifierDict["values"]

    nImages = min(len(dataset), 1600)
    batchSize = 16
    nRuns = int(nImages / batchSize)

    r = 0
    accuracy = 0

    keyOrders = dataset.getKeyOrders()[categoryName]
    labelConverter = buildLabelConverted(keyOrders["values"], classifierDict["match"])

    for data, labels in dbLoader:

        printProgressBar(r, nRuns)

        preds = classifier(data).detach()
        preds = torch.argmax(preds, dim=1)
        convertedLabels = torch.tensor([labelConverter[x] for x in labels], device =torch.device("cuda:0") )

        accuracy += float(torch.sum(preds == convertedLabels).item())

        r +=1
        if r >= nRuns:
            break

    printProgressBar(nRuns, nRuns)

    accuracy /= batchSize * nRuns
    print(accuracy)

    outPath = os.path.join(checkPointDir, name) + "_trainGAN_metric.json"
    saveScore(outPath, accuracy, categoryName)
