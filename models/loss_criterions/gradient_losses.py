# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True):
    r"""
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf

    Args:

        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    batchSize = input.size(0)
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(input.nelement() /
                                        batchSize)).contiguous().view(
                                            input.size())
    alpha = alpha.to(input.device)
    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)

    decisionInterpolate = discriminator(interpolates, False)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()


def logisticGradientPenalty(input, discrimator, weight, backward=True):
    r"""
    Gradient penalty described in "Which training method of GANs actually
    converge
    https://arxiv.org/pdf/1801.04406.pdf

    Args:

        - input (Tensor): batch of real data
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    locInput = torch.autograd.Variable(
        input, requires_grad=True)
    gradients = torch.autograd.grad(outputs=discrimator(locInput)[:, 0].sum(),
                                    inputs=locInput,
                                    create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients = (gradients * gradients).sum(dim=1).mean()

    gradient_penalty = gradients * weight
    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()
