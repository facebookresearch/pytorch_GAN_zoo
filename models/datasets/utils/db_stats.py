def getClassStats(inputDict, className):

    outStats = {}

    for item in inputDict:

        val = item[className]
        if val not in outStats:
            outStats[val] = 0

        outStats[val] += 1

    return outStats


def buildDictStats(inputDict, classList):

    locStats = {"total": len(inputDict)}

    for cat in classList:

        locStats[cat] = getClassStats(inputDict, cat)

    return locStats


def buildKeyOrder(shiftAttrib,
                  shiftAttribVal,
                  stats=None):
    r"""
    If the dataset is labelled, give the order in which the attributes are given

    Args:

        - shiftAttrib (dict): order of each category in the category vector
        - shiftAttribVal (dict): list (ordered) of each possible labels for each
                                category of the category vector
        - stats (dict): if not None, number of representant of each label for
                        each category. Will update the output dictionary with a
                        "weights" index telling how each labels should be
                        balanced in the classification loss.

    Returns:

        A dictionary output[key] = { "order" : int , "values" : list of string}
    """

    MAX_VAL_EQUALIZATION = 10

    output = {}
    for key in shiftAttrib:
        output[key] = {}
        output[key]["order"] = shiftAttrib[key]
        output[key]["values"] = [None for i in range(len(shiftAttribVal[key]))]
        for cat, shift in shiftAttribVal[key].items():
            output[key]["values"][shift] = cat

    if stats is not None:
        for key in output:

            n = sum([x for key, x in stats[key].items()])

            output[key]["weights"] = {}
            for item, value in stats[key].items():
                output[key]["weights"][item] = min(
                    MAX_VAL_EQUALIZATION, n / float(value + 1.0))

    return output
