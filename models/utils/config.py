# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class BaseConfig():
    r"""
    An empty class used for configuration members
    """

    def __init__(self, orig=None):
        if orig is not None:
            print("cawet")


def getConfigFromDict(obj, inputDict, defaultConfig):
    r"""
    Using a new configuration dictionary and a default configuration
    setup an object with the given configuration.

    for example, if you have
    inputDict = {"coin": 22}
    defaultConfig.coin = 23
    defaultConfig.pan = 12

    Then the given obj will get two new members 'coin' and 'pan' with
    obj.coin = 22
    obj.pan = 12

    Args:

        - obj (Object): the object to modify.
        - inputDict (dictionary): new configuration
        - defaultConfig (Object): default configuration
    """
    if not inputDict:
        for member, value in vars(defaultConfig).items():
            setattr(obj, member, value)
    else:
        for member, value in vars(defaultConfig).items():
            setattr(obj, member, inputDict.get(member, value))


def updateConfig(obj, ref):
    r"""
    Update a configuration with the fields of another given configuration
    """

    if isinstance(ref, dict):
        for member, value in ref.items():
            setattr(obj, member, value)

    else:

        for member, value in vars(ref).items():
            setattr(obj, member, value)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise AttributeError('Boolean value expected.')


def updateParserWithConfig(parser, defaultConfig):

    for name, key in vars(defaultConfig).items():
        if key is None:
            continue

        if isinstance(key, bool):
            parser.add_argument('--' + name, type=str2bool, dest=name)
        else:
            parser.add_argument('--' + name, type=type(key), dest=name)

    parser.add_argument('--overrides',
                        action='store_true',
                        help= "For more information on attribute parameters, \
                        please have a look at \
                        models/trainer/standard_configurations")
    return parser


def getConfigOverrideFromParser(parsedArgs, defaultConfig):

    output = {}
    for arg, value in parsedArgs.items():
        if value is None:
            continue

        if arg in vars(defaultConfig):
            output[arg] = value

    return output


def getDictFromConfig(obj, referenceConfig, printDefault=True):
    r"""
    Retrieve all the members of obj which are also members of referenceConfig
    and dump them into a dictionnary

    If printDefault is activated, members of referenceConfig which are not found
    in obj will also be dumped
    """

    output = {}
    for member, value in vars(referenceConfig).items():
        if hasattr(obj, member):
            output[member] = getattr(obj, member)
        elif printDefault:
            output[member] = value

    return output
