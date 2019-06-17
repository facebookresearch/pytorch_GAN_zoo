# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import json
import math

import torch


def isinf(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `+/-INF` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of
        `+/-INF` elements and 0 otherwise

    Example::

        >>> torch.isinf(torch.Tensor([1, float('inf'), 2,
                            float('-inf'), float('nan')]))
        tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor.abs() == math.inf


def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `NaN` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN`
        elements.

    Example::

        >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
        tensor([ 0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor


def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        infGrads = isinf(p.grad.data)
        p.grad.data[infGrads] = 0

        nanGrads = isnan(p.grad.data)
        p.grad.data[nanGrads] = 0


def prepareClassifier(module, outFeatures):

    model = module()
    inFeatures = model.fc.in_features
    model.fc = torch.nn.Linear(inFeatures, outFeatures)

    return model


def getMinOccurence(inputDict, value, default):

    keys = list(inputDict.keys())
    outKeys = [x for x in keys if x <= value]
    outKeys.sort()

    if len(outKeys) == 0:
        return default

    return inputDict[outKeys[-1]]


def getNameAndPackage(strCode):

    if strCode == 'PGAN':
        return "progressive_gan", "ProgressiveGAN"

    if strCode == 'PPGAN':
        return "pp_gan", "PPGAN"

    if strCode == "DCGAN":
        return "DCGAN", "DCGAN"

    if strCode == "StyleGAN":
        return "styleGAN", "StyleGAN"

    raise ValueError("Unrecognized code " + strCode)


def parse_state_name(path):
    r"""
    Parse a file name with the given pattern:
    pattern = ($model_name)_s($scale)_i($iteration).pt

    Returns: None if the path doesn't fulfill the pattern
    """
    path = os.path.splitext(os.path.basename(path))[0]

    data = path.split('_')

    if len(data) < 3:
        return None

    # Iteration
    if data[-1][0] == "i" and data[-1][1:].isdigit():
        iteration = int(data[-1][1:])
    else:
        return None

    if data[-2][0] == "s" and data[-2][1:].isdigit():
        scale = int(data[-2][1:])
    else:
        return None

    name = "_".join(data[:-2])

    return name, scale, iteration


def parse_config_name(path):
    r"""
    Parse a file name with the given pattern:
    pattern = ($model_name)_train_config.json

    Raise an error if the pattern doesn't match
    """

    path = os.path.basename(path)

    if len(path) < 18 or path[-18:] != "_train_config.json":
        raise ValueError("Invalid configuration path")

    return path[:-18]


def getLastCheckPoint(dir, name, scale=None, iter=None):
    r"""
    Get the last checkpoint of the model with name @param name detected in the
    directory (@param dir)

    Returns:
    trainConfig, pathModel, pathTmpData

    trainConfig: path to the training configuration (.json)
    pathModel: path to the model's weight data (.pt)
    pathTmpData: path to the temporary configuration (.json)
    """
    trainConfig = os.path.join(dir, name + "_train_config.json")

    if not os.path.isfile(trainConfig):
        return None

    listFiles = [f for f in os.listdir(dir) if (
        os.path.splitext(f)[1] == ".pt" and
        parse_state_name(f) is not None and
        parse_state_name(f)[0] == name)]

    if scale is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[1] == scale]

    if iter is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[2] == iter]

    listFiles.sort(reverse=True, key=lambda x: (
        parse_state_name(x)[1], parse_state_name(x)[2]))

    if len(listFiles) == 0:
        return None

    pathModel = os.path.join(dir, listFiles[0])
    pathTmpData = os.path.splitext(pathModel)[0] + "_tmp_config.json"

    if not os.path.isfile(pathTmpData):
        return None

    return trainConfig, pathModel, pathTmpData


def getVal(kwargs, key, default):

    out = kwargs.get(key, default)
    if out is None:
        return default

    return out


def toStrKey(item):

    if item is None:
        return ""

    out = "_" + str(item)
    out = out.replace("'", "")
    return out


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def loadPartOfStateDict(module, state_dict, forbiddenLayers=None):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    if forbiddenLayers is None:
        forbiddenLayers = []
    for name, param in state_dict.items():
        if name.split(".")[0] in forbiddenLayers:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        own_state[name].copy_(param)


def loadStateDictCompatible(module, state_dict):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        if name in own_state:
            own_state[name].copy_(param)
            continue

        # Else see if the input name is a prefix
        suffixes = ["bias", "weight"]
        found = False
        for suffix in suffixes:
            indexEnd = name.find(suffix)
            if indexEnd > 0:
                newKey = name[:indexEnd] + "module." + suffix
                if newKey in own_state:
                    own_state[newKey].copy_(param)
                    found = True
                    break

        if not found:
            raise AttributeError("Unknow key " + name)


def loadmodule(package, name, prefix='..'):
    r"""
    A dirty hack to load a module from a string input

    Args:
        package (string): package name
        name (string): module name

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from " + prefix + package + " import " + name + " as module"
    exec(strCmd)
    return eval('module')


def saveScore(outPath, outValue, *args):

    flagPath = outPath + ".flag"

    while os.path.isfile(flagPath):
        time.sleep(1)

    open(flagPath, 'a').close()

    if os.path.isfile(outPath):
        with open(outPath, 'rb') as file:
            outDict = json.load(file)
        if not isinstance(outDict, dict):
            outDict = {}
    else:
        outDict = {}

    fullDict = outDict

    for item in args[:-1]:
        if str(item) not in outDict:
            outDict[str(item)] = {}
        outDict = outDict[str(item)]

    outDict[args[-1]] = outValue

    with open(outPath, 'w') as file:
        json.dump(fullDict, file, indent=2)

    os.remove(flagPath)
