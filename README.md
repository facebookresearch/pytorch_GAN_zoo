# SL_fashionGen
Generation of fashion items using GANs

Pytorch training code for generating images of clothing items. It is based on [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) and [CAN](https://arxiv.org/pdf/1706.07068.pdf) (Creative Adversarial Networks) code and papers.

## Requirements

This project requires:
- the very last version of pytorch
- numpy
- scipy

Optional:
- visdom

Fair setup:

Before running the project on the FAIR cluster, don't forget to setup the environment

```
module load NCCL/2.2.13-cuda.9.0 && module load anaconda3 && source activate fair_env_latest_py3
```

## Repository architecture

### internship

Othman's internship code + tex report

### models

GAN models implementation

### stab_stats_maker.py

A script used to estimate the importance of the different configuration's
parameter for the gan stability.

### build_classifier.py

A script used to build classifiers for the inception and am score.

## How to run a test ?

Go to the main directory and tape the command

```
python tests.py {names of the tests you want to run}
```

All test are listed in models/test.

For example you can run

```
python tests.py test_save
```

## How to run a training session ?

```
python train.py $MODEL_NAME -c $CONFIGURATION_FILE -n $RUN_NAME [OVERRIDES]
```

See base_config.json for an example of a typical configuration file for progressive GANs. Don't forget to change the path of your input dataset in the configuration file !

Your run will be saved in testNets\$RUN_NAME. If a checkpoint is detected in this directory the training will restart from the last detected checkpoint. Please use the option --restart to force the training to start over.

The following models are available for training:
PGAN: progressive gan
PPGAN: progressive gan with a product architecture

OVERRIDES : you can override the input configuration (see below) by specifying new values in the command line. Typically:

```
python train.py PPGAN -c coin.json -n PAN --learningRate 0.2
```

Will force the learning rate to be 0.2 in the training whatever the configuration file coin.json specifies.

To get all the possible overrides, please type:

```
python train.py $MODEL_NAME --overrides
```

### Mandatory fields in your configuration file:
pathDB : path to the directory where the training dataset is saved

### Optional fields of the configuration file:
config: new configuration. Must be a dictionary. If a field is left empty then the default value will be used. Please refer to models/trainer/std_p_gan_config.py to have a detailed description of the possible fields with progressive gan and models/trainer/std_p_gan_config.py for the product progressive gan.

Load an existing checkpoint:
  - checkpointData : a dictionary with tree entries
    - pathTrainConfig : path to the json configuration of the checkpoint
    - pathModel: path to the networks (.pt file)
    - pathTmpConfig: path to the temporary data of the training (.json)

    For example a typical checkpoint would be
    "checkpointData":{"pathTrainConfig": "celeba_check_train_config.json",
                      "pathModel": "celeba_check_s2_i20000.pt",
                      "pathTmpConfig": "celeba_check_s2_i20000_tmp_config.json"}

Add labels to the dataset:
  - pathAttrib : path to the .json file matching each image name with its attributes
  - selectedAttributes : if only some attributes should be considered, list them here. ex ["texture", "shape"]
  - ignoreAttribs : if set to true, all attributes will be ignored and the attribute dictionary will be used only as a way to filter the input dataset.

Change the configuration at some scales:
  - configScheduler : a dictionary. Each key is a scale index ("0" -> 4x4, "1" -> 8x8 etc..), associated to a configuration dictionary with the fields to change
  - miniBatchScheduler : a dictionary. Each key is a scale key associated to the new mini batch size from this scale. Ex: {"2":128, "5":16}, then from scales 2 to 4 the mini batch size will be 128, and from scale 5 it will be 16.

Other options:
 - celebaHQDB : set to True if you want to train on celeba HD (1024x1024)
 - imagefolderDataset : set to True if your input dataset is saved in this format https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder

## How to run a training session for progressive gan with the product architecture ?
```
python tests.py test_train_pppgan -c $CONFIGURATION_FILE -n $RUN_NAME
```

The configuration file is the same as for progressive gan with a few more possible options
- pathDBMask : if you want to add a shape discrimator in your model, add here the path to the mask database. The match between an image and its mask should be done as follow: $MASK_NAME = $IMAGE_NAME + "_mask.jpg" ___

## How to run a evaluation of the results of your training session ?

You need to use the eval.py script.

### Image generation

You can generate more images from an existing checkpoint using:
```
python eval.py visualization -n $modelName -m $modelType
```

Where modelType is in [PGAN, PPGAN, DCGAN] and modelName is the name given to your model. This script will load the last checkpoint detected at testNets/$modelName. If you want to load a specific iteration, please call:

```
python eval.py visualization -n $modelName -m $modelType -s $SCALE -i $ITER
```

If your model is conditioned, you can ask the visualizer to print out some conditioned generations. For example:

```
python eval.py visualization -n $modelName -m $modelType --Class T_SHIRT
```

Will plot a series of T_SHIRTS in visdom. Please use the option --showLabels to see all the available labels for your model.

### SWD metric

Using the same kind of configuration file as above, just launch:

```
python eval.py laplacian_SWD -c $CONFIGURATION_FILE -n $modelName -m $modelType
```
Where $CONFIGURATION_FILE is the training configuration file called by train.py (see above)
You can add optional arguments:
-s $SCALE : specify the scale at which the evaluation should be done (if not set, take the highest one)
-i $ITER : specify the iteration to evaluate (if not set, all checkpoints will be evaluated in descending order)

### Inspirational generation

To make an inspirational generation

```
python eval.py inspirational_generation -n $modelName -m $modelType --inputImage $pathTotheInputImage [-f $pathToTheFeatureExtractor]
```

## I have generated my metrics. How can i plot them on visdom ?

Just run
```
python eval.py metric_plot -n $modelName
```

## cpp/ : C++ code for datasets transformations

You will find here the C++ code that extracts the masks from the YSL dataset

To extract the masks from the YSL dataset.

1) compile the MaskMaker binary as shown in cpp/README.md
2) Update mask_extraction.py with the relevant output paths and launch the script
