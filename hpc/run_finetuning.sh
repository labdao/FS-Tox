#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

module load anaconda3/personal
source ~/.bashrc
conda activate fs-tox

cd /rds/general/user/ssh22/home/FS-Tox

python src/models/finetune.py