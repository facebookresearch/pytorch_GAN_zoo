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
export inspire="dtd20"
export inspire="celeba"
python nevergrad/Test_inspiration-celebA.py | tee rescelebauls_`date | sed 's/ /_/g'`

if [ "$inspire" == "celeba" ]; then
./viewer_celeb_camille.sh
fi
if [ "$inspire" == "dtd20" ]; then
./viewer.sh
fi




