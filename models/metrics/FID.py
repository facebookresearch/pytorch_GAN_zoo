import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class FIDScore():
    def __init__(self,
                 device=None):

        if device is None:
            device = torch.device('cpu')

        modules = list(models.inception_v3(pretrained=True).children())
        outDim = modules[-1].in_features
        self.classifier = nn.Sequential(*modules[:-1])

        self.meanReal = torch.zeros(outDim, 1, device=device)
        self.meanFake = torch.zeros(outDim, 1, device=device)

        self.covarReal = torch.zeros(outDim, outDim, device=device)
        self.covarFake = torch.zeros(outDim, outDim, device=device)

        self.nReal = 0
        self.nFake = 0

    def updateWithMiniBatch(self, real, fake):

        nReal = featuresReal.size(0)
        nFake = featuresFake.size(0)

        featuresReal = self.classifier(real).view(nReal, -1, 1)
        featuresFake = self.classifier(fake).view(nFake, -1, 1)

        self.meanReal += featuresReal.sum(dim=0)
        self.meanFake += featuresFake.sum(dim=0)

        self.covarReal += (featuresReal *
                           featuresReal.transpose(1, 2)).sum(dim=0)
        self.covarFake += (featuresFake *
                           featuresFake.transpose(1, 2)).sum(dim=0)

        self.nReal += nReal
        self.nFake += nFake

    def getScore(self):

        self.meanReal /= float(self.nReal)
        self.meanFake /= float(self.nFake)
        self.covarReal /= float(self.nReal)
        self.covarFake /= float(self.nFake)

        self.covarReal -= self.meanReal*self.meanReal.transpose(1, 2)
        self.covarFake -= self.meanFake*self.meanFake.transpose(1, 2)

        return calculate_frechet_distance(self.meanReal.cpu().numpy(),
                                          self.covarReal.cpu().numpy(),
                                          self.meanFake.cpu().numpy(),
                                          self.covarFake.cpu().numpy())
