# Pytorch GAN Zoo

Several GAN implementations:
- Progressive Growiing of GAN (PGAN): https://arxiv.org/pdf/1710.10196.pdf
- Decoupled progressive growing (PPGAN)
- DCGAN: https://arxiv.org/pdf/1511.06434.pdf (incoming)

## Requirements

This project requires:
- pytorch (fair_env_latest_py3 version)
- numpy
- scipy

Optional:
- visdom

FAIR setup:

Before running the project on the FAIR cluster, don't forget to setup the environment

```
module load NCCL/2.2.13-cuda.9.0 && module load anaconda3 && source activate fair_env_latest_py3
```

## Quick training

If you want to waste no time and just launch a training session on celeba cropped

```
python setup.py celeba_cropped $PATH_TO_CELEBA/img_align_celeba/ -o $OUTPUT_DATASET
python train.py PGAN -c config_celeba_cropped.json --restart -n celeba_cropped
```

And wait for a few days. Your checkpoints will be dumped in output_networks/celeba_cropped. You should get 128x128 generations at the end.

For celebaHQ:

```
python setup.py celeba_cropped $PATH_TO_CELEBAHQ -o $OUTPUT_DATASET -f
python train.py PGAN -c config_celebaHQ.json --restart -n celebaHQ
```

Your checkpoints will be dumped in output_networks/celebaHQ. You should get 1024x1024 generations at the end.

## Advanced guidelines

### How to run a training session ?

```
python train.py $MODEL_NAME -c $CONFIGURATION_FILE [-n $RUN_NAME] [-d $OUTPUT_DIRECTORY] [OVERRIDES]
```

Where:

1 - MODEL_NAME is the name of the model you want to run. Currently, two models are available:
  - PGAN (progressive growing of gan)
  - PPGAN (decoupled version of PGAN)
2 - CONFIGURATION_FILE (mandatory): path to a training configuration file. This file is a json file containing at least a pathDB entry with the path to the training dataset. See below for more informations about this file.
3 - RUN_NAME is the name you want to give to your training session. All checkpoints will be saved in $OUTPUT_DIRECTORY/$RUN_NAME. Default value is default
4 - OUTPUT_DIRECTORY is the directory were all training sessions are saved. Default value is output_networks
5 - OVERRIDES: you can overrides some of the models parameters defined in the configuration file in the command line. For example:

```
python train.py PPGAN -c coin.json -n PAN --learningRate 0.2
```

Will force the learning rate to be 0.2 in the training whatever the configuration file coin.json specifies.

To get all the possible overrides, please type:

```
python train.py $MODEL_NAME --overrides
```

### Configuration file of a training session

The minimum necessary file for a training session is a json with the following lines

```
{
  "pathDB": PATH_TO_YOUR_DATASET
}
```

Where a dataset can be:
- a folder with all your images in .jpg, .png or .npy format
- a folder with N subfolder and images in it
- a .h5 file (cf fashionGen)

To this you can add a "config" entry giving overrides to the standard configuration. See models/trainer/standard_configurations to see all possible overrides. For example:

```
{
  "pathDB": PATH_TO_YOUR_DATASET,
  "config":{"baseLearningRate":0.1,
            "miniBatchSize":22}
}
```

Will override the learning rate and the mini-batch-size.

Other fields are available on the configuration file, like:
- pathAttribDict (string): path to a .json file matching each image with its attributes
- selectedAttributes (list): if specified, learn only the given attributes during the training session
- pathDBMask (string): for decoupled models, path of the mask database. The match between an image and its mask should be done as follow: $MASK_NAME = $IMAGE_NAME + "_mask.jpg"__
- pathPartition (string): path to a partition of the training dataset
- partitionValue (string): if pathPartition is specified, name of the partition to choose
- miniBatchScheduler (dictionary): dictionary updating the size of the mini batch at different scale of the training
                                  ex {"2":16, "7":8} meaning that the mini batch size will be 16 from scale 16 to 6 and 8 from scale 7
- configScheduler (dictionary): dictionary updating the model configuration at different scale of the training
                                ex {"2":{"baseLearningRate":0.1, "epsilonD":1}} meaning that the learning rate and epsilonD will be updated to 0.1 and 1 from scale 2 and beyond

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
