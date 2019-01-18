import argparse
import json
import os

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
                        celeba_cropped, celebaHQ')
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

    if args.dataset_name not in ['celeba', 'celeba_cropped', 'celebaHQ']:
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
