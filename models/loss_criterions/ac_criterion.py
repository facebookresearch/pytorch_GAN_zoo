from math import *
from copy import deepcopy
from random import randint

import sys

import torch
import torch.nn.functional as F

import numpy as np


class ACGanCriterion:
    r"""
    Class implementing all tools necessary for a GAN to take into account class
    conditionning while generating a model (cf Odena's AC-GAN)
    """

    def __init__(self,
                 attribKeysOrder,
                 allowMultiple=None):
        r"""
        Args:

            attribKeysOrder (dict): dictionary containing the all the possible
                                    categories of the DCGAN model and their order.

                                    Each entry of the attribKeysOrder is another
                                    dictionary with two fields:

                                    order: order of the category in the input vector
                                    values: possible values taken by this category

                                    Such a dictionary is returned by
                                    models.datasets.attrib_dataset.AttribDataset.getKeyOrders()
            Ex:
                attribKeysOrder = {"Gender": {"order": 0, "values":["M", "W"]},
                                  "Nationality": {"order": 1, "values":["english", "french", "indian"]}
                                   }
                allowMultiple = ["Nationality"]

                Then a category vector corresponding to this pair could be:
                V = [0, 1, 1, 1, 0]

                Which would correspond to a sample of gender "W" and nationalities
                "english" and "french"
        """
        self.nAttrib = len(attribKeysOrder)
        self.attribSize = [0 for i in range(self.nAttrib)]
        self.keyOrder = ['' for x in range(self.nAttrib)]
        self.labelsOrder = {}

        self.inputDict = deepcopy(attribKeysOrder)

        for key in attribKeysOrder:
            order = attribKeysOrder[key]["order"]
            self.keyOrder[order] = key
            self.attribSize[order] = len(attribKeysOrder[key]["values"])
            self.labelsOrder[key] = {index: label for label, index in enumerate(
                attribKeysOrder[key]["values"])}

        self.labelWeights = torch.tensor(
            [1.0 for x in range(self.getInputDim())])

        for key in attribKeysOrder:
            order = attribKeysOrder[key]["order"]
            if attribKeysOrder[key].get('weights', None) is not None:
                shift = sum(self.attribSize[:order])

                for value, weight in attribKeysOrder[key]['weights'].items():
                    self.labelWeights[shift +
                                      self.labelsOrder[key][value]] = weight

        self.allowMultiple = [False for i in range(self.nAttrib)]

        if allowMultiple:
            for item in allowMultiple:
                self.allowMultiple[attribKeysOrder[item]["order"]] = True

        self.sizeOutput = sum([self.attribSize[i] if self.allowMultiple[i] else 1
                               for i in range(self.nAttrib)])

        self.normalizationFactor = 1.0

    def generateConstraintsFromVector(self, n, labels):

        vect = []

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            key = self.keyOrder[i]

            if key in labels:
                value = labels[key]
                index = self.labelsOrder[key][value]
                out = torch.zeros(n, C, 1, 1)
                out[:, index] = 1
            else:
                v = np.random.randint(0, C, n)
                w = np.zeros((n, C), dtype='float32')
                w[np.arange(n), v] = 1
                out = torch.tensor(w).view(n, C, 1, 1)

            vect.append(out)
        return torch.cat(vect, dim=1)

    def getAttribShift(self, index):

        if index == 0:
            return 0
        return sum(self.attribSize[:index])

    def getAllAttribShift(self):

        tot = len(self.attribSize)
        return [sum(self.attribSize[:x]) for x in range(tot)]

    def buildRandomCriterionTensor(self, sizeBatch):
        r"""
        Build a batch of vectors with a random combination of the values of the
        existing classes

        Args:
            sizeBatch (int): number of vectors to generate

        Return:
            targetVector, latentVector

            targetVector : [sizeBatch, M] tensor used as a reference for the loss
                           computation (see self.getLoss)
            latentVector : [sizeBatch, M', 1, 1] tensor. Should be concatenatenated
                          with the random GAN input latent vcetor

            M' > M, input latent data should be coded with one-hot inputs while
            pytorch requires a different format for softmax loss (see self.getLoss)
        """
        targetOut = []
        inputLatent = []

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            if self.allowMultiple[i]:
                y = torch.randint(0, 2, (sizeBatch, C))
                targetOut.append(y)
                inputLatent.append(y.view(sizeBatch, C))

            else:
                v = np.random.randint(0, C, sizeBatch)
                w = np.zeros((sizeBatch, C), dtype='float32')
                w[np.arange(sizeBatch), v] = 1
                y = torch.tensor(w).view(sizeBatch, C)

                inputLatent.append(y)
                targetOut.append(torch.tensor(v).float().view(sizeBatch, 1))

        return torch.cat(targetOut, dim=1), torch.cat(inputLatent, dim=1)

    def getInputDim(self):
        r"""
        Size of the latent vector given by self.buildRandomCriterionTensor
        """
        return sum(self.attribSize)

    def getLoss(self, output, target):
        r"""
        Compute the conditional loss between the network's output and the target.
        This loss, L, is the sum of the losses Lc of the categories defined in
        the criterion. We have:

                 | Cross entropy loss for the class c if c is attached to a classification task
            Lc = | Multi label soft margin loss for the class c if c is attached to a tagging task
        """
        loss = 0
        shiftInput = 0
        shiftTarget = 0

        self.labelWeights = self.labelWeights.to(output.device)

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            locInput = output[:, shiftInput:(shiftInput+C)]

            if self.allowMultiple[i]:
                locTarget = target[:, shiftTarget:(shiftTarget+C)]
                locLoss = F.multilabel_soft_margin_loss(locInput, locTarget)
                shiftTarget += C
            else:
                locTarget = target[:, shiftTarget]
                locLoss = F.cross_entropy(locInput, locTarget.long(
                ), weight=self.labelWeights[shiftInput:(shiftInput+C)])
                shiftTarget += 1

            loss += locLoss
            shiftInput += C

        return loss * self.normalizationFactor

    def getPredictionVector(self, rawValues, doClip=True):

        shiftInput = 0

        output = []
        outLabels = []

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            locInput = rawValues[:, shiftInput:(shiftInput+C)]
            locPred = F.softmax(locInput, dim=1)

            if doClip:
                tmp = torch.argmax(locPred, dim=1, keepdim=False)
                pred = torch.zeros(locPred.size(), device=rawValues.device)
                pred[:, tmp] = 1

                output.append(pred)

                className = self.keyOrder[i]
                classLabel = self.inputDict[className]["values"][tmp.item()]

                print(classLabel)

            else:

                output.append(locPred)

        return torch.cat(output, dim=1)
