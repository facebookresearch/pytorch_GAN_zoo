# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import torch
import h5py

import copy

from .utils.db_stats import buildKeyOrder


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 file_path,
                 partition_path=None,
                 partition_value=None,
                 transform=None,
                 specificAttrib=None,
                 stats_file=None,
                 pathDBMask=None):
        super(H5Dataset, self).__init__()

        self.path = file_path
        self.partition_path = partition_path
        self.partition_value = partition_value

        if self.partition_value is None:
            self.partition_path = None
            print("No partition value found, ignoring the partition file")

        self.h5_file = None
        self.partition_file = None

        self.transform = transform

        self.attribKeys = copy.deepcopy(specificAttrib)
        self.statsData = None
        self.totAttribSize = 0

        if stats_file is not None:
            with open(stats_file, 'rb') as file:
                self.statsData = json.load(file)

            if self.partition_value is None and "GLOBAL" in self.statsData:
                self.statsData = self.statsData["GLOBAL"]

            elif self.partition_value in self.statsData:
                self.statsData = self.statsData[self.partition_value]

            self.buildAttribShift()

        self.pathDBMask = pathDBMask
        self.maskFile = None

    def __getitem__(self, index):

        if self.h5_file is None:
            self.h5_file = h5py.File(self.path, 'r')

            if self.partition_path is not None:
                self.partition_file = h5py.File(self.partition_path, 'r')

        if self.partition_file is not None:
            index = self.partition_file[self.partition_value][index]

        img = self.h5_file['input_image'][index]

        if self.transform is not None:
            img = self.transform(img)

        if self.statsData is not None:

            attr = [None for x in range(self.totAttribSize)]

            for key in self.attribKeys:

                label = str(self.h5_file[key][index][0])
                shift = self.attribShift[key]
                attr[shift] = self.attribShiftVal[key][label]

        else:

            attr = [0]

        if self.pathDBMask is not None:

            if self.maskFile is None:
                self.maskFile = h5py.File(self.pathDBMask, 'r')

            mask = self.maskFile["mask"][index]
            mask = self.transform(mask)

            img = img * (mask + 1.0) * 0.5 + (1 - mask) * 0.5

            return img, torch.tensor(attr), mask

        return img, torch.tensor(attr)

    def __len__(self):
        if self.partition_path is None:
            with h5py.File(self.path, 'r') as db:
                lens = len(db['input_image'])
        else:
            with h5py.File(self.partition_path, 'r') as db:
                lens = len(db[self.partition_value])
        return lens

    def getName(self, index):

        if self.partition_path is not None:
            if self.partition_file is None:
                self.partition_file = h5py.File(self.partition_path, 'r')

            return self.partition_file[self.partition_value][index]

        return index

    def buildAttribShift(self):

        self.attribShift = None
        self.attribShiftVal = None

        if self.statsData is None:
            return

        if self.attribKeys is None:
            self.attribKeys = [x for x in self.statsData.keys() if
                               x != "totalSize"]

        self.attribShift = {}
        self.attribShiftVal = {}

        self.totAttribSize = 0

        for key in self.attribKeys:

            self.attribShift[key] = self.totAttribSize
            self.attribShiftVal[key] = {
                name: c
                for c, name in enumerate(list(self.statsData[key].keys()))}
            self.totAttribSize += 1

    def getKeyOrders(self, equlizationWeights=False):

        if equlizationWeights:
            raise ValueError("Equalization weight not implemented yet")

        return buildKeyOrder(self.attribShift, self.attribShiftVal)
