# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import h5py
import pickle

import math
import numpy as np
import imageio

from models.utils.utils import printProgressBar
from models.utils.image_transform import NumpyResize, pil_loader


def saveImage(path, image):
    return imageio.imwrite(path, image)


def unpackCIFAR10(pathDB, pathOut):
    toLoad = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
              'data_batch_5']

    # Check valididty:
    for item in toLoad:
        filePath = os.path.join(pathDB, item)
        if not os.path.isfile(filePath):
            raise FileNotFoundError("Can't find " + filePath)

    if not os.path.isdir(pathOut):
        os.mkdir(pathOut)

    pathLabels = os.path.join(pathDB, 'batches.meta')
    with open(pathLabels, 'rb') as file:
        labels = pickle.load(file)['label_names']

    for label in labels:
        pathOutLabel = os.path.join(pathOut, label)
        if not os.path.isdir(pathOutLabel):
            os.mkdir(pathOutLabel)

    nImagesPerBatch = 10000
    nImages = nImagesPerBatch * len(toLoad)
    status = 0

    print("Unpacking CIFAR-10...")
    for item in toLoad:
        pathItem = os.path.join(pathDB, item)
        with open(pathItem, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        data = dict[str.encode('data')]
        dataLabel = dict[str.encode('labels')]
        dataNames = dict[str.encode('filenames')]

        assert(len(dataLabel) == nImagesPerBatch)
        assert(data.shape[1] == 3072)

        for i in range(nImagesPerBatch):

            rgbArray = np.zeros((32, 32, 3), 'uint8')
            rgbArray[:, :, 0] = data[i, :1024].reshape(32, 32)
            rgbArray[:, :, 1] = data[i, 1024:2048].reshape(32, 32)
            rgbArray[:, :, 2] = data[i, 2048:].reshape(32, 32)

            name = dataNames[i].decode("utf-8")
            label = labels[dataLabel[i]]

            path = os.path.join(pathOut, os.path.join(label, name))
            saveImage(path, rgbArray)

            printProgressBar(status, nImages)
            status += 1

    printProgressBar(nImages, nImages)


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
        # correspond only to miss-labelled data
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
    scales = [0, 5, 6, 8]
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
                        choices=['celeba','celeba_cropped', 'celebaHQ', 'dtd',
                                 'fashionGen', 'cifar10'],
                        help='Name of the dataset.')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the input dataset')
    parser.add_argument('-o', help="If it applies, output dataset (mandadory \
                        for celeba_cropped)",
                        type=str, dest="output_dataset")
    parser.add_argument('-f', action='store_true',
                        dest="fast_training",
                        help="Store several resized versions of a dataset for \
                        a faster training. Advised for HD datasets.")
    parser.add_argument('-m', dest='model_type',
                        type=str, default='PGAN',
                        choices=['PGAN', 'DCGAN'],
                        help="Model type. Default is progressive growing \
                        (PGAN)")

    args = parser.parse_args()

    config = {"pathDB": args.dataset_path}
    config["config"] = {}

    moveLastScale = False
    keepOriginalDataset = True

    if args.dataset_name in ['celeba', 'celeba_cropped']:
        if args.model_type == 'PGAN':
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
        if args.model_type == 'PGAN':
            config["miniBatchScheduler"] = {"7": 12, "8": 8}
        if args.model_type == 'DCGAN':
            print("WARNING: DCGAN is diverging for celebaHQ")

    if args.dataset_name == 'cifar10':
        if args.output_dataset is None:
            raise AttributeError(
                "Please provide and output path to dump cifar10")
        unpackCIFAR10(args.dataset_path, args.output_dataset)
        if args.fast_training:
            print("Ignoring the fast training parameter for cifar10")
            args.fast_training = False

        config["imagefolderDataset"] = True
        config["pathDB"] = args.output_dataset
        if args.model_type == 'PGAN':
            config["config"]["maxIterAtScale"] = [96000, 120000, 120000,
                                                  120000]

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

        config["pathPartition"] = pathPartition
        config["pathAttribDict"] = pathStats
        config["config"]["weightConditionG"] = 1.0
        config["config"]["weightConditionD"] = 1.0

        if args.model_type == 'PGAN':
            config["config"]["maxIterAtScale"] = [48000, 96000, 96000, 96000,
                                                  96000, 96000, 96000]

    if args.dataset_name == 'dtd':

        maxSize = 256
        config["pathDB"] = args.dataset_path
        moveLastScale = False
        config["imagefolderDataset"] = True
        config["config"]["weightConditionG"] = 1.0
        config["config"]["weightConditionD"] = 1.0

        if args.fast_training:
            print("Ignoring the fast training parameter for DTD")
            args.fast_training = False

        if args.model_type == 'PGAN':
            config["config"]["maxIterAtScale"] = [48000, 96000, 96000, 96000,
                                                  96000, 96000, 96000]

    if args.fast_training:
        if args.output_dataset is None:
            raise AttributeError(
                "Please provide and output path to dump intermediate datasets")

        maxScale = int(math.log(maxSize, 2)) - 2
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

            datasetProfile[maxScale] = lastScaleOut
        elif keepOriginalDataset:
            datasetProfile, _ = resizeDataset(
                args.dataset_path, args.output_dataset, maxSize / 2)
            datasetProfile[maxScale] = args.dataset_path
            lastScaleOut = args.dataset_path
        else:
            datasetProfile, lastScaleOut = resizeDataset(
                args.dataset_path, args.output_dataset, maxSize)

        config["datasetProfile"] = datasetProfile
        config["pathDB"] = lastScaleOut

    pathConfig = "config_" + args.dataset_name + ".json"
    with open(pathConfig, 'w') as file:
        json.dump(config, file, indent=2)
