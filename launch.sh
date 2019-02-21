module load anaconda3
module load NCCL/2.2.13-cuda.9.0 
module load anaconda3 
source activate fair_env_latest_py3

pip3 install -r requirements.txt
export PYTHONPATH=/private/home/oteytaud/NEVERGRAD:$PYTHONPATH
export PYTHONPATH=/private/home/oteytaud/morgane/pytorch_GAN_zoo:$PYTHONPATH

python nevergrad/Test_inspiration3.py | tee resuls_`date | sed 's/ /_/g'`




