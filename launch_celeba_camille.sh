module load anaconda3
module load NCCL/2.2.13-cuda.9.0 
module load anaconda3 
source activate fair_env_latest_py3

pip3 install -r requirements.txt
#export PYTHONPATH=/private/home/oteytaud/NEVERGRAD:$PYTHONPATH
export PYTHONPATH=/private/home/coupriec/Riviere2018Fashion/pytorch_GAN_zoo:$PYTHONPATH
export username=`whoami`
rm /private/home/${username}/HDGANSamples/random_gens/*/*.jpg
rm /private/home/${username}/HDGANSamples/random_gens/*.jpg

#export inspire="dtd20"
#export inspire="celeba"
export inspire="dtd20miss"

#export full="full"
export full="limited"

#export loss="vgg"
#export loss="l2"
export loss="mixed"

export renorm="renorm"  #we push z towards limited norm.
#export renorm="none"

export nstep="500"

python nevergrad/Test_inspiration-celebA.py | tee rescelebauls_`date | sed 's/ /_/g'`

./viewer_celeb_camille.sh
