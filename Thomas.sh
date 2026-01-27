#! /bin/bash

#SBATCH --job-name=Test_git_SVD
#SBATCH --output=Open-3DULM-main/Result/ML-%j-open_3D_ulm_main_Thomas.out
#SBATCH --error=Open-3DULM-main/Result/ML-%j-open_3D_ulm_main_Thomas.err


#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.emile@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "Open-3DULM-main/scripts/open_3D_ulm_main.py" --input 3D_ULM_Data --output results