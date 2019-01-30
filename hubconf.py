'''
hubconf.py for pytorch_gan_zoo repo

## Users can get the diverse models of pytorch_gan_zoo by calling
hub_model = hub.load(
    '??/pytorch_gan_zoo:master',
    $MODEL_NAME, #
    pretrained=False) # (Not pretrained models online yet)

Available model'names are [DCGAN, PGAN]

'''

import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch', 'torchvision', 'visdom', 'numpy', 'h5py', 'scipy']


def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from models.progressive_gan import PGAN
    if config not in kwargs:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs['useGPU'],
                 storeAVG=True,
                 **kwargs['config'])

    checkpoint = 'coin'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint))
    return model

def DCGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from models.progressive_gan import PGAN
    if config not in kwargs:
        kwargs['config'] = {}

    model = DCGAN(useGPU=kwargs['useGPU'],
                  storeAVG=True,
                  **kwargs['config'])

    checkpoint = 'coin'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint))
    return model
