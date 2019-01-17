import torchvision
import torchvision.transforms as Transforms

import numpy as np
import scipy
import scipy.misc

# The equivalent of some torchvision.transforms operations but for numpy array
# instead of PIL images

class NumpyResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        r"""
        Args:

            img (np array): image to be resized

        Returns:

            np array: resized image
        """

        return scipy.misc.imresize(img, self.size, interp='bilinear')

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class NumpyFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        random.seed(None)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flip(img, 1).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        r"""
        Turn a numpy objevt into a tensor.
        """

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        return Transforms.functional.to_tensor(img)
