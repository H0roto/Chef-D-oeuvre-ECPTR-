#! /bin/bash

#SBATCH --job-name=Run_train_yolo
#SBATCH --output=Open-3DULM-main/Result/ML-%j-YOLO_train_Dylan.out
#SBATCH --error=Open-3DULM-main/Result/ML-%j-YOLO_train_Dylan.err

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

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "YOLO/training_YOLO.py" --model "YOLO/yolo12s.pt" --data "YOLO/dataset_yolo_15dB/dataset.yaml" --project "YOLO/YOLO_Results" --name "YOLO12s_Training_15dB_160"