#! /bin/bash

#SBATCH --job-name=Titre
#SBATCH --output=Result/out/transformPAL_PNG_Thomas.out
#SBATCH --error=Result/err/transformPAL_PNG_Thomas.err


#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.emile@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "testRF/transforme_PALA-mat_PNG.ipnyb" --input 3D_ULM_Data --output results