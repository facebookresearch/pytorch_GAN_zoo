import json
import os
import csv

from db_stats import buildDictStats

def getBaseAttribDict(pathcsv,
                      pathdir,
                      categories = None,
                      imgExt =".jpg"):

    # Step zero: checks
    if categories is not None and 'Filename' not in categories:
        categories.append('Filename')

    # Step one: load the img list
    imgList = [ f for f in os.listdir(pathdir) if os.path.splitext(f)[1] == imgExt]
    imgList.sort()

    # Step two: load the csv file
    with open(pathYSLMetadata, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        if categories is None:
            csvList = [f for f in reader if f['Filename'] in imgList]

        else:
            csvList = [ { cat : f[cat] for cat in categories} \
                        for f in reader \
                        if f['Filename'] in imgList]

    return csvList

def castAttribMapping(inputAttrib, inputCat, inputMapping):

    for item in inputAttrib:
        item[inputCat] = inputMapping.get(item[inputCat], None)

    inputAttrib = [f for f in inputAttrib if f[inputCat] is not None]

def buildAttribDict(pathcsv,
                    pathdir,
                    attribsToKeep,
                    attribsToMatch):

    for attrib in attribsToMatch.keys():
        if attrib not in attribsToKeep:
            attribsToKeep.append(attrib)

    baseDict = getBaseAttribDict(pathcsv, pathdir, attribsToKeep)

    for match, pathMatch in attribsToMatch.items():

        with open(pathMatch, 'rb') as file:
            matchDict = json.load(file)
            castAttribMapping(baseDict, match, matchDict)

    return baseDict

def filterByCat(inputDict, discriminatingCat):

    outputDict = []

    for f in inputDict:
        toTake = True
        for cat in discriminatingCat:
            if f[cat] not in discriminatingCat[cat]:
                toTake = False

        if toTake:
            outputDict.append(f)
    return outputDict

def addExtension(imgName, extension):
    return os.path.splitext(imgName)[0] + extension + os.path.splitext(imgName)[1]


if __name__ == "__main__":

    basePathYSL = "/private/home/sbaio/YSL_2Aug/Saint Laurent Packshot "
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019"]
    pathBaseout = "/private/home/mriviere/YSL"

    dirOut = "/private/home/mriviere/YSL/"
    nameOut = "clothes_mask"

    categoriesToFilter = { 'Department' : "2018_Department_simplified.json",
                            'Class' : "2018_Class_simplified.json",
                            'View' : "2018_View_simplified.json"}
                            #'Colour' : "2018_Colour_simplified.json"}

    for key in categoriesToFilter:
        categoriesToFilter[key] = os.path.join("/private/home/mriviere/", categoriesToFilter[key])

    attribsToKeep = ['Filename', 'Year', 'Season']

    allCategories = [ f for f in categoriesToFilter.keys()]  + attribsToKeep[1:]

    outDict = []

    discriminativeCat = {"Class": ["COAT", "SHIRT", "JACKET", "DRESS", "PANTS",
                        "SKIRT", "T_SHIRT", "SHORT", "JUMPSUIT"],
                        "View": ["FRONT", "BACK"], "Department": ["MEN", "WOMEN"]}

    for year in years:
        pathYSL = basePathYSL + year
        pathYSLMetadata = os.path.join(pathYSL, "_metadata packshot " + year  + ".csv")

        locDict = buildAttribDict(pathYSLMetadata, pathYSL, attribsToKeep, categoriesToFilter)
        locDict = filterByCat(locDict, discriminativeCat)

        outDict = outDict + locDict

    fullStats = buildDictStats(outDict, allCategories)

    pathOut = os.path.join(dirOut, nameOut + "_stats.json")

    with open(pathOut, 'w') as file:
        json.dump(fullStats, file, indent = 2)

    pathOut = os.path.join(dirOut, nameOut + "_dict.json")
    with open(pathOut, 'w') as file:
        outDict = {addExtension(f['Filename'],'_mask') : {x : f[x] for x in f if x is not 'Filename'} for f in outDict}
        json.dump(outDict, file, indent = 2)
