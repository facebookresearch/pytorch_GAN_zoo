# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import importlib
import argparse

from models.utils.utils import getVal, getLastCheckPoint, loadmodule
from models.utils.config import getConfigOverrideFromParser, \
    updateParserWithConfig

import json


def getTrainer(name):

    match = {"PGAN": ("progressive_gan_trainer", "ProgressiveGANTrainer"),
             "StyleGAN":("styleGAN_trainer", "StyleGANTrainer"),
             "DCGAN": ("DCGAN_trainer", "DCGANTrainer")}

    if name not in match:
        raise AttributeError("Invalid module name")

    return loadmodule("models.trainer." + match[name][0],
                      match[name][1],
                      prefix='')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        help='Name of the model to launch, available models are\
                        PGAN, PPGAN adn StyleGAN. To get all possible option for a model\
                         please run train.py $MODEL_NAME -overrides')
    parser.add_argument('--no_vis', help=' Disable all visualizations',
                        action='store_true')
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based \
                        visualizer (SLURM)',
                        action='store_true')
    parser.add_argument('--restart', help=' If a checkpoint is detected, do \
                                           not try to load it',
                        action='store_true')
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name", default="default")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default='output_networks')
    parser.add_argument('-c', '--config', help="Model's name",
                        type=str, dest="configPath")
    parser.add_argument('-s', '--save_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="saveIter", default=16000)
    parser.add_argument('-e', '--eval_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved",
                        type=int, dest="evalIter", default=100)
    parser.add_argument('-S', '--Scale_iter', help="If it applies, scale to work\
                        on")
    parser.add_argument('-v', '--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")

    # Retrieve the model we want to launch
    baseArgs, unknown = parser.parse_known_args()
    trainerModule = getTrainer(baseArgs.model_name)

    # Build the output durectory if necessary
    if not os.path.isdir(baseArgs.dir):
        os.mkdir(baseArgs.dir)

    # Add overrides to the parser: changes to the model configuration can be
    # done via the command line
    parser = updateParserWithConfig(parser, trainerModule._defaultConfig)
    kwargs = vars(parser.parse_args())
    configOverride = getConfigOverrideFromParser(
        kwargs, trainerModule._defaultConfig)

    if kwargs['overrides']:
        parser.print_help()
        sys.exit()

    # Checkpoint data
    modelLabel = kwargs["name"]
    restart = kwargs["restart"]
    checkPointDir = os.path.join(kwargs["dir"], modelLabel)
    checkPointData = getLastCheckPoint(checkPointDir, modelLabel)

    if not os.path.isdir(checkPointDir):
        os.mkdir(checkPointDir)

    # Training configuration
    configPath = kwargs.get("configPath", None)
    if configPath is None:
        raise ValueError("You need to input a configuratrion file")

    with open(kwargs["configPath"], 'rb') as file:
        trainingConfig = json.load(file)

    # Model configuration
    modelConfig = trainingConfig.get("config", {})
    for item, val in configOverride.items():
        modelConfig[item] = val
    trainingConfig["config"] = modelConfig

    # Visualization module
    vis_module = None
    if baseArgs.np_vis:
        vis_module = importlib.import_module("visualization.np_visualizer")
    elif baseArgs.no_vis:
        print("Visualization disabled")
    else:
        vis_module = importlib.import_module("visualization.visualizer")

    print("Running " + baseArgs.model_name)

    # Path to the image dataset
    pathDB = trainingConfig["pathDB"]
    trainingConfig.pop("pathDB", None)

    partitionValue = getVal(kwargs, "partition_value",
                            trainingConfig.get("partitionValue", None))

    GANTrainer = trainerModule(pathDB,
                               useGPU=True,
                               visualisation=vis_module,
                               lossIterEvaluation=kwargs["evalIter"],
                               checkPointDir=checkPointDir,
                               saveIter= kwargs["saveIter"],
                               modelLabel=modelLabel,
                               partitionValue=partitionValue,
                               **trainingConfig)

    # If a checkpoint is found, load it
    if not restart and checkPointData is not None:
        trainConfig, pathModel, pathTmpData = checkPointData
        print(f"Model found at path {pathModel}, pursuing the training")
        GANTrainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)

    GANTrainer.train()
