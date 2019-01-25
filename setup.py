import argparse
import json
import os
import h5py

import numpy as np
from scipy import misc

from models.utils.utils import printProgressBar
from models.utils.image_transform import NumpyResize, pil_loader


def saveImage(path, image):
    return misc.imsave(path, image)


def celebaSetup(inputPath,
                outputPath,
                pathConfig="config_celeba_cropped.json"):

    imgList = [f for f in os.listdir(
        inputPath) if os.path.splitext(f)[1] == ".jpg"]
    cx = 89
    cy = 121

    nImgs = len(imgList)

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    for index, item in enumerate(imgList):
        printProgressBar(index, nImgs)
        path = os.path.join(inputPath, item)
        img = np.array(pil_loader(path))

        img = img[cy - 64: cy + 64, cx - 64: cx + 64]

        path = os.path.join(outputPath, item)
        saveImage(path, img)

    printProgressBar(nImgs, nImgs)


def fashionGenSetup(fashionGenPath,
                    outputPath):

    basePath = os.path.splitext(outputPath)[0]

    if not os.path.isdir(basePath):
        os.mkdir(basePath)

    outputPath = os.path.join(basePath, os.path.basename(basePath))

    h5file = h5py.File(fashionGenPath)
    imgKey = 'input_image'
    validClasses = ["input_gender", "input_category", "input_pose"]
    nImgs = h5file[imgKey].shape[0]

    outIndexes = {}
    statsPartition = {"GLOBAL": {"input_department": {},
                                 "totalSize": 0}
                      }

    for attribute in validClasses:
        statsPartition["GLOBAL"][attribute] = {}

    partitionCategory = "input_department"

    print("Building the partition..")

    for index in range(nImgs):

        rawVal = str(h5file[partitionCategory][index][0])
        val = rawVal.replace("b'", "").replace("'", "")
        strVal = str(val)

        # Hand-made fix for the clothing dataset : some pose attributes
        # corresponds only to miss-labelled data
        if strVal == "CLOTHING" \
                and str(h5file["input_pose"][index][0]) in \
                ["b'id_gridfs_6'", "b'id_gridfs_5'"]:
            continue

        if strVal not in statsPartition:
            statsPartition[strVal] = {attribute: {}
                                      for attribute in validClasses}
            statsPartition[strVal]["totalSize"] = 0
            outIndexes[val] = []
            statsPartition["GLOBAL"]["input_department"][rawVal] = 0

        outIndexes[val].append(index)
        statsPartition[strVal]["totalSize"] += 1
        statsPartition["GLOBAL"]["input_department"][rawVal] += 1
        statsPartition["GLOBAL"]["totalSize"] += 1

        for attribute in validClasses:

            label = str(h5file[attribute][index][0])

            if label not in statsPartition[strVal][attribute]:
                statsPartition[strVal][attribute][label] = 0
            if label not in statsPartition["GLOBAL"][attribute]:
                statsPartition["GLOBAL"][attribute][label] = 0

            statsPartition[strVal][attribute][label] += 1
            statsPartition["GLOBAL"][attribute][label] += 1

        printProgressBar(index, nImgs)
    printProgressBar(nImgs, nImgs)

    h5file.close()

    pathPartition = outputPath + "_partition.h5"
    f = h5py.File(pathPartition, 'w')

    for key, value in outIndexes.items():
        f.create_dataset(key, data=np.array(value))
    f.close()

    pathStats = outputPath + "_stats.json"
    with open(pathStats, 'w') as file:
        json.dump(statsPartition, file, indent=2)

    return pathPartition, pathStats


def resizeDataset(inputPath, outputPath, maxSize):

    sizes = [64, 128, 512, 1024]
    scales = [0, 5, 7, 8]
    index = 0

    imgList = [f for f in os.listdir(inputPath) if os.path.splitext(f)[
        1] in [".jpg", ".npy"]]

    nImgs = len(imgList)

    if maxSize < sizes[0]:
        raise AttributeError("Maximum resolution too low")

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    datasetProfile = {}

    for index, size in enumerate(sizes):

        if size > maxSize:
            break

        localPath = os.path.join(outputPath, str(size))
        if not os.path.isdir(localPath):
            os.mkdir(localPath)

        datasetProfile[str(scales[index])] = localPath

        print("Resolution %d x %d" % (size, size))

        resizeModule = NumpyResize((size, size))

        for index, item in enumerate(imgList):
            printProgressBar(index, nImgs)
            path = os.path.join(inputPath, item)
            img = pil_loader(path)

            img = resizeModule(img)
            path = os.path.splitext(os.path.join(localPath, item))[0] + ".jpg"
            saveImage(path, img)

        printProgressBar(nImgs, nImgs)

    return datasetProfile, localPath


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('dataset_name', type=str,
                        help='Name of the dataset. Available options: celeba, \
                        celeba_cropped, celebaHQ, fashionGen')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the input dataset')
    parser.add_argument('-o', help="If it applies, output dataset (mandadory \
                        for celeba_cropped)",
                        type=str, dest="output_dataset")
    parser.add_argument('-f', action='store_true',
                        dest="fast_training",
                        help="Store several resized versions of a dataset for \
                        a faster training. Advised for HD datasets.")

    args = parser.parse_args()

    config = {"pathDB": args.dataset_path}
    config["config"] = {}

    moveLastScale = False
    keepOriginalDataset = True

    if args.dataset_name not in ['celeba', 'celeba_cropped', 'celebaHQ',
                                 'fashionGen']:
        raise AttributeError(args.dataset_name + " unknown datatset")

    if args.dataset_name in ['celeba', 'celeba_cropped']:
        config["config"]["maxIterAtScale"] = [48000, 96000, 96000, 96000,
                                              96000, 96000]
        maxSize = 128

    if args.dataset_name == 'celeba_cropped':
        if args.output_dataset is None:
            raise AttributeError(
                "Please provide and output path to dump celebaCropped")

        print("Cropping dataset...")
        celebaSetup(args.dataset_path, args.output_dataset)
        config["pathDB"] = args.output_dataset
        moveLastScale = True

    if args.dataset_name == 'celebaHQ':
        maxSize = 1024
        moveLastScale = False
        keepOriginalDataset = True

    if args.dataset_name == 'fashionGen':

        if args.output_dataset is None:
            raise AttributeError(
                "Please provide and output path to dump the fashionGen \
                partition")

        if args.fast_training:
            print("Ignoring the fast training parameter for fashionGen")
            args.fast_training = False

        pathPartition, pathStats = fashionGenSetup(args.dataset_path,
                                                   args.output_dataset)

        config = {"pathPartition": pathPartition,
                  "pathAttribDict": pathStats,
                  "pathDB": args.dataset_path,
                  "config": {
                      "maxIterAtScale": [20000, 40000, 40000,
                                         40000, 40000, 80000,
                                         80000],
                      "weightConditionG": 1.0,
                      "weightConditionD": 1.0}
                  }

    if args.fast_training:
        if args.output_dataset is None:
            raise AttributeError(
                "Please provide and output path to dump intermediate datasets")

        if moveLastScale:
            datasetProfile, _ = resizeDataset(
                args.output_dataset, args.output_dataset, maxSize / 2)

            print("Moving the last dataset...")

            lastScaleOut = os.path.join(args.output_dataset, str(maxSize))
            if not os.path.isdir(lastScaleOut):
                os.mkdir(lastScaleOut)

            for img in [f for f in os.listdir(args.output_dataset)
                        if os.path.splitext(f)[1] == ".jpg"]:
                pathIn = os.path.join(args.output_dataset, img)
                pathOut = os.path.join(lastScaleOut, img)

                os.rename(pathIn, pathOut)

            datasetProfile[str(maxSize)] = lastScaleOut
        elif keepOriginalDataset:
            datasetProfile, _ = resizeDataset(
                args.dataset_path, args.output_dataset, maxSize / 2)
            datasetProfile[str(maxSize)] = args.dataset_path
            lastScaleOut = args.dataset_path
        else:
            datasetProfile, lastScaleOut = resizeDataset(
                args.dataset_path, args.output_dataset, maxSize)

        config["datasetProfile"] = datasetProfile
        config["pathDB"] = lastScaleOut

    pathConfig = "config_" + args.dataset_name + ".json"
    with open(pathConfig, 'w') as file:
        json.dump(config, file, indent=2)
