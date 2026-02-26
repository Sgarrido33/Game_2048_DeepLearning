#!/bin/bash
#SBATCH --job-name=2048_datagen
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=08:00:00        
#SBATCH --output=logs_gen_%j.txt

module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate animal-env

SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH

echo "Iniciando generación de partidas"
cd /home/rodrigo.castro/2048

python generate_data.py --episodes 20000 --cores 30 --out_dir $SCRATCH

echo "Borrando archivos crudos en Scratch."
rm -rf $SCRATCH
echo "Generación finalizada. Data en dataset.7z"
