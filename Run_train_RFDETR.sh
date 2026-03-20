#! /bin/bash

#SBATCH --job-name=train_RFDETR
#SBATCH --output=Result/out/rfderRun.out
#SBATCH --error=Result/err/rfdetrRun.err


#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.emile@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX6000Node
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

set -euo pipefail

cd /projects/ecptr/Espace_GIT/Espace_Dylan

SCRIPT="/projects/ecptr/Espace_GIT/Espace_Dylan/Chef-D-oeuvre-ECPTR-/train_rfdetr.py"

echo "PWD=$(pwd)"
echo "SCRIPT=$SCRIPT"
ls -l "$SCRIPT"

srun singularity exec --nv /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif \
  torchrun --standalone --nproc_per_node=1 \
  "$SCRIPT"