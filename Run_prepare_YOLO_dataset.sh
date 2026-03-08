#! /bin/bash

#SBATCH --job-name=Run_prepare_YOLO_dataset
#SBATCH --output=Open-3DULM-main/Result/ML-%j-YOLO_prepare_dataset_Dylan.out
#SBATCH --error=Open-3DULM-main/Result/ML-%j-YOLO_prepare_dataset_Dylan.err

#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paquie@univ-tlse3.fr

#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paquie@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
# #SBATCH --mem=80G

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "YOLO/prepare_yolo_dataset.py" --input "YOLO/PALA_data_InSilicoFlow/IQ" --output "YOLO/dataset/dataset_yolo" --box-size 5 --noise-levels 10 15 20
