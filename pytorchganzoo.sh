
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
fi

PATH_TO_CELEBA=/private/home/mriviere/celebaHQ/512

OUTPUT_DATASET=celeba_output

CONFIGURATION_FILE=config_celeba_cropped.json

#python3.6 datasets.py celeba_cropped $PATH_TO_CELEBA -o $OUTPUT_DATASET


alphaJumpMode linear|custom
initBiasToZero True|False
perChannelNormalization True|False
#leakyness 0.2
#epsilonD 0.001
miniBatchStdDev False|True
#baseLearningRate 0.001


lr=`./paramcloseto0.sh 0.00001 0.1 $1`  # first parameter for learning rate
ed=`./paramcloseto0.sh 0.00001 0.1 $2`  # first parameter for epsilond
ln=`./paramcloseto0.sh 0.01 0.6 $3`  # first parameter for epsilond
python3.6 train.py PGAN -c $CONFIGURATION_FILE  --restart -n celeba_output --no_vis --baseLearningRates $lr --epsilonD $ed --leakyness $ln
# #Options:
# #--learning_rate XXXXX
# 
# python eval.py laplacian_SWD -c $CONFIGURATION_FILE -n $modelName -m $modelType

