# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json

from ..utils.utils import getVal


def test(parser, visualisation=None):

    # Parameters
    kwargs = vars(parser.parse_args())

    name = getVal(kwargs, "name", None)
    if name is None:
        raise ValueError("You need to input a name")

    if visualisation is None:
        raise ValueError("A visualizer is mandatory for this evaluation")

    checkPointDir = os.path.join(kwargs["dir"], name)

    suffixes = {"SWD": "_swd", "NN": "_nn_metric",
                "INCEPTION": "_inception_metric"}

    for key, value in suffixes.items():

        pathFile = os.path.join(checkPointDir, name + value + ".json")

        if not os.path.isfile(pathFile):
            continue

        with open(pathFile, 'rb') as file:
            data = json.load(file)

        for scale in data:

            itemType = next(iter(data[scale].values()))

            if isinstance(itemType, dict):

                attribs = list(itemType.keys())
                withAttribs = True
                nData = len(next(iter(itemType.values())))

            else:
                attribs = ['']
                withAttribs = False
                nData = len(next(iter(data[scale].values())))

            for attrib in attribs:

                env_name = name + "_" + key + "_scale_" + \
                    scale + "_" + os.path.basename(attrib)
                visualisation.delete_env(env_name)

                locIter = []
                outYData = [[] for x in range(nData)]

                iterations = [int(x) for x in data[scale].keys() if x.isdigit()]

                iterations.sort()

                for iteration in iterations:

                    locIter.append(iteration)

                    if not withAttribs:

                        for i in range(nData):

                            outYData[i].append(data[scale][str(iteration)][i])

                    else:

                        if attrib not in data[scale][str(iteration)]:
                            continue

                        for i in range(nData):
                            outYData[i].append(
                                data[scale][str(iteration)][attrib][i])

                for i in range(nData):
                    plotName = key + " " + str(i)
                    visualisation.publishLinePlot([(plotName, outYData[i])], locIter,
                                                  name=plotName,
                                                  env=env_name)

                    print(scale, plotName, sum(outYData[i]) / len(outYData[i]))
