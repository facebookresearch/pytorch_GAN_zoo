# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import sys

import torch

from ..gan_visualizer import GANVisualizer
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, \
    getNameAndPackage, parse_state_name


def getModelName(pathConfig):

    pathConfig = os.path.basename(pathConfig)

    if pathConfig[-18:] != '_train_config.json':
        raise ValueError("Invalid configuration name")

    return pathConfig[:-18]


def updateParserWithLabels(parser, labels):

    for key in labels:
        parser.add_argument('--' + key, type=str,
                            help=str(labels[key]["values"]))
    return parser


def test(parser, visualisation=None):

    # Parameters
    parser.add_argument('--showLabels', action='store_true',
                        help="For labelled datasets, show available labels")
    parser.add_argument('--interpolate', type=str,
                        dest='interpolationPath',
                        help="Path to some latent vectors to interpolate")
    parser.add_argument('--random_interpolate', action='store_true',
                        help="Save a random interpolation")
    parser.add_argument('--save_dataset', type=str, dest="output_dataset",
                        help="Save a dataset at the given location")
    parser.add_argument('--size_dataset', type=int, dest="size_dataset",
                        default=10000,
                        help="Size of the dataset to be saved")

    kwargs = vars(parser.parse_known_args()[0])

    name = getVal(kwargs, "name", None)
    if name is None:
        parser.print_help()
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        parser.print_help()
        raise ValueError("You need to input a module")

    scale = getVal(kwargs, "scale", None)
    iter = getVal(kwargs, "iter", None)

    checkPointDir = os.path.join(kwargs["dir"], name)
    checkpointData = getLastCheckPoint(checkPointDir,
                                       name,
                                       scale=scale,
                                       iter=iter)

    if checkpointData is None:
        raise FileNotFoundError(
            "Not checkpoint found for model " + name + " at directory " + dir)

    modelConfig, pathModel, _ = checkpointData
    if scale is None:
        _, scale, _ = parse_state_name(pathModel)

    keysLabels = None
    with open(modelConfig, 'rb') as file:
        keysLabels = json.load(file)["attribKeysOrder"]
    if keysLabels is None:
        keysLabels = {}

    parser = updateParserWithLabels(parser, keysLabels)

    kwargs = vars(parser.parse_args())

    if kwargs['showLabels']:
        parser.print_help()
        sys.exit()

    interpolationPath = getVal(kwargs, 'interpolationPath', None)

    pathLoss = os.path.join(checkPointDir, name + "_losses.pkl")
    pathOut = os.path.splitext(pathModel)[0] + "_fullavg.jpg"

    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)
    exportMask = module in ["PPGAN"]

    visualizer = GANVisualizer(
        pathModel, modelConfig, modelType, visualisation)

    if interpolationPath is None and not kwargs['random_interpolate']:
        nImages = (256 // 2**(max(scale - 2, 3))) * 8
        visualizer.exportVisualization(pathOut, nImages,
                                       export_mask=exportMask)

    toPlot = {}
    for key in keysLabels:
        if kwargs.get(key, None) is not None:
            toPlot[key] = kwargs[key]

    if len(toPlot) > 0:
        visualizer.generateImagesFomConstraints(
            16, toPlot, env=name + "_pictures")

    interpolationVectors = None
    if interpolationPath is not None:
        interpolationVectors = torch.load(interpolationPath)
        pathOut = os.path.splitext(interpolationPath)[0] + "_interpolations"
    elif kwargs['random_interpolate']:
        interpolationVectors, _ = visualizer.model.buildNoiseData(3)
        pathOut = os.path.splitext(pathModel)[0] + "_interpolations"

    if interpolationVectors is not None:

        if not os.path.isdir(pathOut):
            os.mkdir(pathOut)

        nImgs = interpolationVectors.size(0)
        for img in range(nImgs):

            indexNext = (img + 1) % nImgs
            path = os.path.join(pathOut, str(img) + "_" + str(indexNext))

            if not os.path.isdir(path):
                os.mkdir(path)

            path = os.path.join(path, "")

            visualizer.saveInterpolation(
                100, interpolationVectors[img],
                interpolationVectors[indexNext], path)

    outputDatasetPath = getVal(kwargs, "output_dataset", None)
    if outputDatasetPath is not None:
        print("Exporting a fake dataset at path " + outputDatasetPath)
        visualizer.exportDB(outputDatasetPath, kwargs["size_dataset"])

    visualizer.plotLosses(pathLoss, name)
