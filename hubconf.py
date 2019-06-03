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

inputRandom, randomLabels = model.buildNoiseData((int) $BATCH_SIZE)

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

models/trainer/standard_configurations to see all the
training parameters you can use.

'''

import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch']


def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    model_name (string): if pretrained, load one of the following models
    celebaHQ-256, celebaHQ-512, DTD, celeba, cifar10. Default is celebaHQ.
    """
    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs.get('useGPU', True),
                 storeAVG=True,
                 **kwargs['config'])

    checkpoint = {"celebAHQ-256": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ_s6_i80000-6196db68.pth',
                  "celebAHQ-512": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth',
                  "DTD": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/testDTD_s5_i96000-04efa39f.pth',
                  "celeba": "https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s5_i83000-2b0acc76.pth"}
    if pretrained:
        if "model_name" in kwargs:
            if kwargs["model_name"] not in checkpoint.keys():
                raise ValueError("model_name should be in "
                                    + str(checkpoint.keys()))
        else:
            print("Loading default model : celebaHQ-256")
            kwargs["model_name"] = "celebAHQ-256"
        model.load_state_dict(model_zoo.load_url(
                                            checkpoint[kwargs["model_name"]]))
    return model


def DCGAN(pretrained=False, *args, **kwargs):
    """
    DCGAN basic model
    pretrained (bool): load a pretrained model ? In this case load a model
    trained on fashionGen cloth
    """
    from models.DCGAN import DCGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = DCGAN(useGPU=kwargs.get('useGPU', True),
                  storeAVG=True,
                  **kwargs['config'])

    checkpoint = 'https://dl.fbaipublicfiles.com/gan_zoo/DCGAN_fashionGen-1d67302.pth'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint))
    return model
