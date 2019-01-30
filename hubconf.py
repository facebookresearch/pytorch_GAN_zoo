'''
hubconf.py for pytorch_gan_zoo repo

## Users can get the diverse models of pytorch_gan_zoo by calling
hub_model = hub.load(
    '??/pytorch_gan_zoo:master',
    $MODEL_NAME, #
    config = None,
    useGPU = True,
    pretrained=False) # (Not pretrained models online yet)

Available model'names are [DCGAN, PGAN].
The config option should be a dictionnary defining the training parameters of
the model. See ??/pytorch_gan_zoo/models/trainer/standard_configurations to see
all possible options

## How can I use my model ?

### Build a random vector

inputRandom = model.buildRandomCriterionTensor((int) $BATCH_SIZE)

### Feed a random vector to the model

model.test(inputRandom,
           getAvG=True,
           toCPU=True)

Arguments:
    - getAvG (bool) get the smoothed version of the generator (advised)
    - toCPU (bool) if set to False the output tensor will be a torch.cuda tensor

### Acces the generator

model.netG()

### Acces the discriminator

model.netD()

## Can I train my model ?

Of course. You can set all training parameters in the constructor (losses to use,
learning rate, number of iterations etc...) and use the optimizeParameters()
method to make a training steps.

Typically here will be a sample code:

for input_real in dataset:

    allLosses = model.optimizeParameters(inputs_real)

    # Do something with the losses

Please have a look at

??/pytorch_gan_zoo/models/trainer/standard_configurations to see all the
training parameters you can use.

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
    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
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
    from models.progressive_gan import ProgressiveGAN as DCGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = DCGAN(useGPU=kwargs['useGPU'],
                  storeAVG=True,
                  **kwargs['config'])

    checkpoint = 'coin'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint))
    return model
