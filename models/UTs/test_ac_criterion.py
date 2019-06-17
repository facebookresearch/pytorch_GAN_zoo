# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from math import exp, log
import torch
from ..loss_criterions.ac_criterion import ACGanCriterion


def test():

    # test
    attribKeysList = {"Gender": {"order": 0,
                                 "values": ['M', 'W', 'Racoon']
                                 },
                      "Console": {"order": 1,
                                  "values": ["None", "PC", "PS", "XBOX"]
                                  }
                      }
    allowMultiple = ["Console"]
    test = ACGanCriterion(attribKeysList, allowMultiple=allowMultiple)
    tar, inLat = test.buildRandomCriterionTensor(2)

    if tar.size()[0] != 2:
        print("Invalid batch size for the target")
        return False
    if inLat.size()[0] != 2:
        print("Invalid batch size for the input latent vector")
        return False
    if tar.size()[1] != 5:
        print("Invalid feature size for the target")
        return False
    if inLat.size()[1] != 7:
        print("Invalid feature size for the input latent vector")
        return False

    testTarget = torch.tensor([[0., 1., 0., 0., 1.],
                               [2., 0., 1., 0., 0.]])
    testTensor = torch.tensor([[0.2, 0.1, 0.7, 0.5, 0.8, 0.9, 0.01],
                               [0.2, 0.1, 0.7, 0.5, 0.8, 0.9, 0.01]])

    a = -0.2 + log(exp(0.2) + exp(0.1) + exp(0.7))
    b = -0.7 + log(exp(0.2) + exp(0.1) + exp(0.7))
    c = log(1 + exp(-0.5)) - log(exp(-0.8)/(1+exp(-0.8))) \
        - log(exp(-0.9)/(1+exp(-0.9))) + log(1 + exp(-0.01))
    d = - log(exp(-0.5)/(1+exp(-0.5))) + log(1 + exp(-0.8)) \
        - log(exp(-0.9)/(1+exp(-0.9))) - log(exp(-0.01)/(1 + exp(-0.01)))

    expectedResult = (a+b+(c+d)/4)/2
    result = test.getLoss(testTensor, testTarget)

    assert abs(result.item() - expectedResult) <= 0.001
