import os
import json
import sys
import torch
import torchvision
from ..trainer.pp_gan_trainer import PPGANTrainer
from ..utils.utils import getVal, getLastCheckPoint
from ..utils.config import getConfigOverrideFromParser, updateParserWithConfig

def train(parser, visualization = None):

    parser = updateParserWithConfig(parser, PPGANTrainer._defaultConfig)

    kwargs = vars(parser.parse_args())
    configOverride = getConfigOverrideFromParser(kwargs, PPGANTrainer._defaultConfig)

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
    modelConfig        = wholeConfig.get("config", {})

    for item, val in configOverride.items():
        modelConfig[item] = val

    # Optional fields
    pathAttrib         = wholeConfig.get("pathAttrib", None)
    selectedAttributes = wholeConfig.get("selectedAttributes", None)
    miniBatchScheduler = wholeConfig.get("miniBatchScheduler", None)
    ignoreAttribs      = wholeConfig.get("ignoreAttribs", False)
    pathDBMask         = wholeConfig.get("pathDBMask", None)
    datasetProfile     = wholeConfig.get("datasetProfile", None)
    maskProfile        = wholeConfig.get("maskProfile", None)
    pathPartition      = wholeConfig.get("pathPartition", None)
    partitionValue     = wholeConfig.get("partitionValue", None)
    configScheduler    = wholeConfig.get("configScheduler", None)

    partitionValue     = getVal(kwargs, "partition_value", partitionValue)
    pathAttrib         = getVal(kwargs, "statsFile", pathAttrib)

    # Script
    modelLabel         = getVal(kwargs, "name", "default")
    checkPointDir      = getVal(kwargs, "dir", os.path.join('testNets', modelLabel))
    lossIterEvaluation = getVal(kwargs, "evalIter", 100)
    saveIter           = getVal(kwargs, "saveIter", 1000)
    restart            = getVal(kwargs, "restart", False)

    if not os.path.isdir(checkPointDir):
        os.mkdir(checkPointDir)

    trainer2 = PPGANTrainer(path_db,
                            useGPU             = True,
                            visualisation      = visualization,
                            lossIterEvaluation = lossIterEvaluation,
                            checkPointDir      = checkPointDir,
                            saveIter           = saveIter,
                            config             = modelConfig,
                            modelLabel         = modelLabel,
                            pathAttribDict     = pathAttrib,
                            specificAttrib     = selectedAttributes,
                            pathDBMask         = pathDBMask,
                            miniBatchProfile   = miniBatchScheduler,
                            datasetProfile     = datasetProfile,
                            maskProfile        = maskProfile,
                            pathPartition      = pathPartition,
                            configScheduler    = configScheduler,
                            partitionValue     = partitionValue)

    checkpointPaths = getLastCheckPoint(checkPointDir, modelLabel)

    if not restart and checkpointPaths is not None:

        trainConfig, pathModel, pathTmpData = checkpointPaths
        trainer2.loadSavedTraining(pathModel, trainConfig, pathTmpData)

    elif "checkpointData" in wholeConfig:

        checkpointData = wholeConfig["checkpointData"]
        if "checkpointDiscriminator" in "checkpointData":
            trainer2.initializeWithPretrainNetworks(checkpointData["checkpointDiscriminator"],
                                                    checkpointData["checkpointGShape"],
                                                    checkpointData["checkpointGTexture"],
                                                    "",
                                                    finetune = False)

            if "shapeDiscrimator" in checkpointData:
                trainer2.model.loadShapeDiscriminator(checkpointData["shapeDiscrimator"])

    trainer2.train()
