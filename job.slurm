#!/bin/bash
#SBATCH --job-name=2048_train_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs_%j.txt

module purge
module load miniconda/3.0 cuda/12.6
eval "$(conda shell.bash hook)"
conda activate animal-env

SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH
DATASET_ORIGEN="/home/rodrigo.castro/2048/dataset.7z"

echo "Descomprimiendo datos en $SCRATCH..."
python -m py7zr x $DATASET_ORIGEN $SCRATCH

echo "Iniciando entrenamiento..."
cd /home/rodrigo.castro/2048
python train.py --data_dir $SCRATCH --output_dir $SLURM_SUBMIT_DIR --epochs 15

echo "Borrando temporales en Scratch."
rm -rf $SCRATCH
echo "Job finalizado."
