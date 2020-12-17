# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torchvision.transforms as Transforms

import os
import random
import numpy as np
import torch
import albumentations as A
import cv2

from PIL import Image



# The equivalent of some torchvision.transforms operations but for numpy array
# instead of PIL images
class ToTensorV2(A.BasicTransform):
    """Convert image and mask to `torch.Tensor`.
    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """
    
    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask
    
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}
    
    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)
    
    def get_transform_init_args_names(self):
        return ("transpose_mask",)
    
    def get_params_dependent_on_targets(self, params):
        return {}

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
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.resize(self.size, resample=Image.BILINEAR))

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


#def pil_loader(path):
#    imgExt = os.path.splitext(path)[1]
#    if imgExt == ".npy":
#        img = np.load(path)[0]
#        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
#
#    # open path as file to avoid ResourceWarning
#    # (https://github.com/python-pillow/Pillow/issues/835)
#    with open(path, 'rb') as f:
#        img = Image.open(f)
#        return img.convert('RGB')

def pil_loader(path, return_np=None):
    if return_np is None:
        return_np = False
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if return_np:
            return np.array(img.convert('RGB'))
        else:
            return img.convert('RGB')

def standardTransform(size):
    return Transforms.Compose([NumpyResize(size),
                               Transforms.ToTensor(),
                               Transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])
