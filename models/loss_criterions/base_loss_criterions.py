# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F


class BaseLossWrapper:
    r"""
    Loss criterion class. Must define 4 members:
    sizeDecisionLayer : size of the decision layer of the discrimator

    getCriterion : how the loss is actually computed

    !! The activation function of the discriminator is computed within the
    loss !!
    """

    def __init__(self, device):
        self.device = device

    def getCriterion(self, input, status):
        r"""
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss

        Args:

            - input (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        pass


class MSE(BaseLossWrapper):
    r"""
    Mean Square error loss.
    """

    def __init__(self, device):
        self.generationActivation = F.tanh
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        size = input.size()[0]
        value = float(status)
        reference = torch.tensor([value]).expand(size, 1).to(self.device)
        return F.mse_loss(F.sigmoid(input[:, :self.sizeDecisionLayer]),
                          reference)


class WGANGP(BaseLossWrapper):
    r"""
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self, device):

        self.generationActivation = None
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        if status:
            return -input[:, 0].sum()
        return input[:, 0].sum()


class Logistic(BaseLossWrapper):
    r"""
    "Which training method of GANs actually converge"
    https://arxiv.org/pdf/1801.04406.pdf
    """

    def __init__(self, device):

        self.generationActivation = None
        self.sizeDecisionLayer = 1
        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        if status:
            return F.softplus(-input[:, 0]).mean()
        return F.softplus(input[:, 0]).mean()


class DCGAN(BaseLossWrapper):
    r"""
    Cross entropy loss.
    """

    def __init__(self, device):

        self.generationActivation = F.tanh
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        size = input.size()[0]
        value = int(status)
        reference = torch.tensor(
            [value], dtype=torch.float).expand(size).to(self.device)
        return F.binary_cross_entropy(torch.sigmoid(input[:, :self.sizeDecisionLayer]), reference)
