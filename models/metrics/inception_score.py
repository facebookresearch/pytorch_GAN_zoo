import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

class InceptionScore():
    def __init__(self,
                 classifier = None,
                 pathPredictions = None,
                 verbose=True):

        self.verbose = verbose
        self.classifier = None

        self.sumEntropy = 0
        self.sumSoftMax = None

        self.nItems = 0

        if classifier:
            self.classifier = classifier.eval()
        elif pathPredictions:
            self.batchPredictions = [torch.load(pathPredictions)]
        else:
            raise ValueError("A classifier or a path to the predictions should \
                             be given")

    def updateWithMiniBatch(self, ref, **kwargs):
        y = self.classifier(ref).detach()

        if self.sumSoftMax is None:
            self.sumSoftMax = torch.zeros(y.size()[1]).to(ref.device)

        # Entropy
        x = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        self.sumEntropy += x.sum().item()

        # Sum soft max
        self.sumSoftMax += F.softmax(y, dim=1).sum(dim=0)

        #N items
        self.nItems += y.size()[0]

    def getScore(self):

        x = self.sumSoftMax
        x = x * torch.log(x / self.nItems)

        output = self.sumEntropy - (x.sum().item())
        output /= self.nItems
        return math.exp(output)
