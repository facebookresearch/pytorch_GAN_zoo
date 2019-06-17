# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

import numpy as np
from ..utils.utils import printProgressBar


def getDescriptorsForMinibatch(minibatch, patchSize, nPatches):
    r"""
    Extract @param nPatches randomly chosen of size patchSize x patchSize
    from each image of the input @param minibatch

    Returns:

        A tensor of SxCxpatchSizexpatchSize where
        S = minibatch.size()[0] * nPatches is the total number of patches
        extracted from the minibatch.
    """
    S = minibatch.size()

    maxX = S[2] - patchSize
    maxY = S[3] - patchSize

    baseX = torch.arange(0, patchSize, dtype=torch.long).expand(S[0] * nPatches,
                                                                patchSize) \
        + torch.randint(0, maxX, (S[0] * nPatches, 1), dtype=torch.long)
    baseY = torch.arange(0, patchSize, dtype=torch.long).expand(S[0] * nPatches,
                                                                patchSize) \
        + torch.randint(0, maxY, (S[0] * nPatches, 1), dtype=torch.long)

    baseX = baseX.view(S[0], nPatches, 1, patchSize).expand(
        S[0], nPatches, patchSize, patchSize)
    baseY = S[2] * baseY.view(S[0], nPatches, patchSize, 1)
    baseY = baseY.expand(S[0], nPatches, patchSize, patchSize)

    coords = baseX + baseY
    coords = coords.view(S[0], nPatches, 1, patchSize, patchSize).expand(
        S[0], nPatches, S[1], patchSize, patchSize)
    C = torch.arange(0, S[1], dtype=torch.long).view(
        1, S[1]).expand(nPatches * S[0], S[1])*S[2]*S[3]
    coords = C.view(S[0], nPatches, S[1], 1, 1) + coords
    coords = coords.view(-1)

    return (minibatch.contiguous().view(-1)[coords]).view(-1, S[1],
                                                          patchSize, patchSize)


def getMeanStdDesc(desc):
    r"""
    Get the mean and the standard deviation of each channel accross the input
    batch.
    """
    S = desc.size()
    assert len(S) == 4
    mean = torch.sum(desc.view(S[0], S[1], -1),
                     dim=2).sum(dim=0) / (S[0] * S[3] * S[2])
    var = torch.sum(
        (desc*desc).view(S[0], S[1], -1), dim=2).sum(dim=0) / \
        (S[0] * S[3] * S[2])
    var -= mean*mean
    var = var.clamp(min=0).sqrt().view(1, S[1]).expand(S[0], S[1])
    mean = (mean.view(1, S[1])).expand(S[0], S[1])

    return mean.view(S[0], S[1], 1, 1), var.view(S[0], S[1], 1, 1)


# -------------------------------------------------------------------------------
# Laplacian pyramid generation, with LaplacianSWDMetric.convolution as input,
# matches the corresponding openCV functions
# -------------------------------------------------------------------------------

def pyrDown(minibatch, convolution):
    x = torch.nn.ReflectionPad2d(2)(minibatch)
    return convolution(x)[:, :, ::2, ::2].detach()


def pyrUp(minibatch, convolution):
    S = minibatch.size()
    res = torch.zeros((S[0], S[1], S[2] * 2, S[3] * 2),
                      dtype=minibatch.dtype).to(minibatch.device)
    res[:, :, ::2, ::2] = minibatch
    res = torch.nn.ReflectionPad2d(2)(res)
    return 4 * convolution(res).detach()

# ----------------------------------------------------------------------------


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    r"""
    NVIDIA's approximation of the SWD distance.
    """
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    results = []
    for repeat in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)
        # normalize descriptor components for each direction
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
        dirs = dirs.astype(np.float32)
        # (neighborhood, direction)
        projA = np.matmul(A, dirs)
        projB = np.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA = np.sort(projA, axis=0)
        projB = np.sort(projB, axis=0)
        # pointwise wasserstein distances
        dists = np.abs(projA - projB)
        # average over neighborhoods and directions
        results.append(np.mean(dists))
    return np.mean(results).item()


def sliced_wasserstein_torch(A, B, dir_repeats, dirs_per_repeat):
    r"""
    NVIDIA's approximation of the SWD distance.
    """
    results = []
    for repeat in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = torch.randn(A.size()[1], dirs_per_repeat,
                           device=A.device, dtype=torch.float32)
        # normalize descriptor components for each direction
        dirs /= torch.sqrt(torch.sum(dirs*dirs, 0, keepdim=True))
        # (neighborhood, direction)
        projA = torch.matmul(A, dirs)
        projB = torch.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA = torch.sort(projA, dim=0)[0]
        projB = torch.sort(projB, dim=0)[0]
        # pointwise wasserstein distances
        dists = torch.abs(projA - projB)
        # average over neighborhoods and directions
        results.append(torch.mean(dists).item())
    return sum(results) / float(len(results))


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


class LaplacianSWDMetric:
    r"""
    SWD metrics used on patches extracted from laplacian pyramids of the input
    images.
    """

    def __init__(self,
                 patchSize,
                 nDescriptorLevel,
                 depthPyramid):
        r"""
        Args:
            patchSize (int): side length of each patch to extract
            nDescriptorLevel (int): number of patches to extract at each level
                                    of the pyramid
            depthPyramid (int): depth of the laplacian pyramid
        """
        self.patchSize = patchSize
        self.nDescriptorLevel = nDescriptorLevel
        self.depthPyramid = depthPyramid

        self.descriptorsRef = [[] for x in range(depthPyramid)]
        self.descriptorsTarget = [[] for x in range(depthPyramid)]

        self.convolution = None

    def updateWithMiniBatch(self, ref, target):
        r"""
        Extract and store decsriptors from the current minibatch
        Args:
            ref (tensor): reference data.
            target (tensor): target data.

            Both tensor must have the same format: NxCxWxD
            N: minibatch size
            C: number of channels
            W: with
            H: height
        """
        target = target.to(ref.device)
        modes = [(ref, self.descriptorsRef), (target, self.descriptorsTarget)]

        assert(ref.size() == target.size())

        if not self.convolution:
            self.initConvolution(ref.device)

        for item, dest in modes:
            pyramid = self.generateLaplacianPyramid(item, self.depthPyramid)
            for scale in range(self.depthPyramid):
                dest[scale].append(getDescriptorsForMinibatch(pyramid[scale],
                                                              self.patchSize,
                                                              self.nDescriptorLevel).cpu().numpy())

    def getScore(self):
        r"""
        Output the SWD distance between both distributions using the stored
        descriptors.
        """
        output = []

        descTarget = [finalize_descriptors(d) for d in self.descriptorsTarget]
        del self.descriptorsTarget

        descRef = [finalize_descriptors(d) for d in self.descriptorsRef]
        del self.descriptorsRef

        for scale in range(self.depthPyramid):
            printProgressBar(scale, self.depthPyramid)
            distance = sliced_wasserstein(
                descTarget[scale], descRef[scale], 4, 128)
            output.append(distance)
        printProgressBar(self.depthPyramid, self.depthPyramid)

        del descRef, descTarget

        return output

    def generateLaplacianPyramid(self, minibatch, num_levels):
        r"""
        Build the laplacian pyramids corresponding to the current minibatch.
        Args:
            minibatch (tensor): NxCxWxD, input batch
            num_levels (int): number of levels of the pyramids
        """
        pyramid = [minibatch]
        for i in range(1, num_levels):
            pyramid.append(pyrDown(pyramid[-1], self.convolution))
            pyramid[-2] -= pyrUp(pyramid[-1], self.convolution)
        return pyramid

    def reconstructLaplacianPyramid(self, pyramid):
        r"""
        Given a laplacian pyramid, reconstruct the corresponding minibatch

        Returns:
            A list L of tensors NxCxWxD, where L[i] represents the pyramids of
            the batch for the ith scale
        """
        minibatch = pyramid[-1]
        for level in pyramid[-2::-1]:
            minibatch = pyrUp(minibatch, self.convolution) + level
        return minibatch

    def initConvolution(self, device):
        r"""
        Initialize the convolution used in openCV.pyrDown() and .pyrUp()
        """
        gaussianFilter = torch.tensor([
            [1, 4,  6,  4,  1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4,  6,  4,  1]], dtype=torch.float) / 256.0

        self.convolution = nn.Conv2d(3, 3, (5, 5))
        self.convolution.weight.data.fill_(0)
        self.convolution.weight.data[0][0] = gaussianFilter
        self.convolution.weight.data[1][1] = gaussianFilter
        self.convolution.weight.data[2][2] = gaussianFilter
        self.convolution.weight.requires_grad = False
        self.convolution = self.convolution.to(device)
