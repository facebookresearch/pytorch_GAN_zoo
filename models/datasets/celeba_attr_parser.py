import json

def parseCeleba(pathCelebaFile):

    with open(pathCelebaFile, 'r') as file:
        data = file.readlines()

    nImg = int(data[0])
    keys = data[1].split()

    nAttrib = len(keys)
    outputAttribDict = {}

    for i in range(nImg):

        currData = data[i + 2].split()
        imgName = currData[0]

        outputAttribDict[imgName] = {}

        for p in range(nAttrib):

            outputAttribDict[imgName][keys[p]] = int((int(currData[p+1]) + 1)/2)

    return outputAttribDict

def parseCelebaHQ(pathCelebaHQFile, celebaHQDict):

    with open(pathCelebaHQFile, 'r') as file:
        data = file.readlines()

    outputDict = {}
    nImg = len(data) - 1

    for i in range(nImg):

        currData = data[i+1].split()
        idx = currData[0]

        originalKey = currData[2]

        nPadding = 5 - len(idx)
        if nPadding > 0:
            idx = "".join(["0" for i in range(nPadding)]) + idx

        idx = "imgHQ" + idx + ".npy"

        outputDict[idx] = celebaHQDict[originalKey]

    return outputDict

pathCelebaFile = "/private/home/mriviere/list_attr_celeba.txt"
pathOutCelebaFile = "/private/home/mriviere/attr_celeba.json"

pathCelebaHQIndex = "/private/home/mriviere/celebaHQ_index.txt"
pathOutCelebaHQFile = "/private/home/mriviere/attr_celebaHQ.json"

outputDict = parseCeleba(pathCelebaFile)

with open(pathOutCelebaFile, 'w') as file:
    json.dump(outputDict, file, indent=2)

celebaHQDict = parseCelebaHQ(pathCelebaHQIndex, outputDict)
with open(pathOutCelebaHQFile, 'w') as file:
    json.dump(celebaHQDict, file, indent=2)
