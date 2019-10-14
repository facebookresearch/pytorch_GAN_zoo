#!/bin/bash

cd /private/home/oteytaud/pytorchganzoo
set -e -x

num_params=3
if [ "$#" -ne $num_params ]; then
    echo "Illegal number of parameters"
    echo num_params
    exit 0
fi

PATH_TO_CELEBAHQ=/private/home/mriviere/celebaHQ/512

OUTPUT_DATASET=celebahq_output$

#CONFIGURATION_FILE=config_celeba_cropped.json
CONFIGURATION_FILE=config_celebaHQ.json

#TODO Run this the first time (and only the first time):
if [ ! -d "$OUTPUT_DATASET" ]; then
  #python3.6 -u datasets.py celeba_cropped $PATH_TO_CELEBAHQ -o $OUTPUT_DATASET
  python3.6 -u datasets.py celebaHQ $PATH_TO_CELEBAHQ -o $OUTPUT_DATASET
fi

# Possible arguments for the training, NOT used in the present script:
#- alphaJumpMode linear|custom
#- initBiasToZero True|False
#- perChannelNormalization True|False
#- miniBatchStdDev False|True

# Possible arguments for the training, used in the present script:
#- leakyness 0.2
#- epsilonD 0.001
#- baseLearningRate 0.001

lr=`./paramcloseto0.sh 0.00001 0.1 $1`  # first parameter for learning rate
ed=`./paramcloseto0.sh 0.00001 0.1 $2`  # first parameter for epsilond
ln=`./paramcloseto0.sh 0.01 0.6 $3`  # first parameter for epsilond

# StyleGAN or DCGAN could replace PGAN.
modelName=PGAN
export NAME=run${RANDOM}_${RANDOM}
python3.6 -u train.py $modelName -c $CONFIGURATION_FILE  --restart -n $NAME --no_vis --baseLearningRate $lr --epsilonD $ed --leakyness $ln --max_time 60 
#for metric in laplacian_SWD inception
for metric in inception
do
python3.6 -u eval.py --no_vis $metric -c $CONFIGURATION_FILE -n $NAME -m $modelName 
done
rm -rf output_networks/$NAME
