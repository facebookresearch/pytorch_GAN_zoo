# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torch.nn.functional as F


class InceptionScore():
    def __init__(self, classifier):

        self.sumEntropy = 0
        self.sumSoftMax = None
        self.nItems = 0
        self.classifier = classifier.eval()

    def updateWithMiniBatch(self, ref):
        y = self.classifier(ref).detach()

        if self.sumSoftMax is None:
            self.sumSoftMax = torch.zeros(y.size()[1]).to(ref.device)

        # Entropy
        x = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        self.sumEntropy += x.sum().item()

        # Sum soft max
        self.sumSoftMax += F.softmax(y, dim=1).sum(dim=0)

        # N items
        self.nItems += y.size()[0]

    def getScore(self):

        x = self.sumSoftMax
        x = x * torch.log(x / self.nItems)
        output = self.sumEntropy - (x.sum().item())
        output /= self.nItems
        return math.exp(output)
