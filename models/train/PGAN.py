import os
import torch
import torchvision
import json
import sys
from ..trainer.progressive_gan_trainer import ProgressiveGANTrainer
from ..utils.utils import getVal, getLastCheckPoint
from ..utils.config import getConfigOverrideFromParser, updateParserWithConfig

def train(parser, visualization = None):

    parser = updateParserWithConfig(parser, ProgressiveGANTrainer._defaultConfig)

    kwargs = vars(parser.parse_args())
    configOverride = getConfigOverrideFromParser(kwargs, ProgressiveGANTrainer._defaultConfig)

    if kwargs['overrides']:
        parser.print_help()
        sys.exit()

    # Parameters
    configPath = kwargs.get("configPath", None)
    if configPath is None:
        raise ValueError("You need to input a configuratrion file")

    with open(kwargs["configPath"], 'rb') as file:
        wholeConfig = json.load(file)

    # Mandatory fields
    path_db            = wholeConfig["pathDB"]

    #Configuration
    modelConfig        = wholeConfig.get("config", {})

    for item, val in configOverride.items():
        modelConfig[item] = val

    # Optional fields
    pathAttrib         = wholeConfig.get("pathAttrib", None)
    selectedAttributes = wholeConfig.get("selectedAttributes", None)
    configScheduler    = wholeConfig.get("configScheduler", None)
    miniBatchScheduler = wholeConfig.get("miniBatchScheduler", None)
    celebaHQDB         = wholeConfig.get("celebaHQDB", False)
    ignoreAttribs      = wholeConfig.get("ignoreAttribs", False)
    imagefolderDataset = wholeConfig.get("imagefolderDataset", False)
    datasetProfile     = wholeConfig.get("datasetProfile", None)
    pathPartition      = wholeConfig.get("pathPartition", None)
    partitionValue     = wholeConfig.get("partitionValue", None)

    partitionValue = getVal(kwargs, "partition_value", None)
    pathAttrib     = getVal(kwargs, "statsFile", pathAttrib)

    if miniBatchScheduler is not None:
        miniBatchScheduler = {int(key): value for key, value \
                              in miniBatchScheduler.items()}

    # To pursue the training of an existing model
    checkpointData     = wholeConfig.get("checkpointData", None)

    # Script
    modelLabel         = getVal(kwargs,"name", "default")
    checkPointDir      = getVal(kwargs, "dir", os.path.join('testNets', modelLabel))
    lossIterEvaluation = getVal(kwargs, "evalIter", 100)
    saveIter           = getVal(kwargs, "saveIter", 1000)
    restart            = getVal(kwargs, "restart", False)

    if os.path.dirname(checkPointDir) == 'testNets' and not os.path.isdir('testNets'):
        os.mkdir(checkPointDir)

    if not os.path.isdir(checkPointDir):
        os.mkdir(checkPointDir)

    trainer2 = ProgressiveGANTrainer(path_db,
                                     useGPU             = True,
                                     visualisation      = visualization,
                                     lossIterEvaluation = lossIterEvaluation,
                                     checkPointDir      = checkPointDir,
                                     saveIter           = saveIter,
                                     config             = modelConfig,
                                     modelLabel         = modelLabel,
                                     celebaHQDB         = celebaHQDB,
                                     pathAttribDict     = pathAttrib,
                                     specificAttrib     = selectedAttributes,
                                     miniBatchProfile   = miniBatchScheduler,
                                     configScheduler    = configScheduler,
                                     ignoreAttribs      = ignoreAttribs,
                                     imagefolderDataset = imagefolderDataset,
                                     datasetProfile     = datasetProfile,
                                     pathPartition      = pathPartition,
                                     partitionValue     = partitionValue)

    checkpointPaths = getLastCheckPoint(checkPointDir, modelLabel)
    if checkpointData is not None:
        trainer2.loadSavedTraining(checkpointData["pathModel"],
                                   checkpointData["pathTrainConfig"],
                                   checkpointData["pathTmpConfig"],
                                   finetune = False)

    elif not restart and checkpointPaths is not None:

        trainConfig, pathModel, pathTmpData = checkpointPaths
        trainer2.loadSavedTraining(pathModel, trainConfig, pathTmpData)


    trainer2.train()
