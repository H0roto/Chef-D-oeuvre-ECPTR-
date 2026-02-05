#! /bin/bash

#SBATCH --job-name=Test_torch_radial_sym
#SBATCH --output=Open-3DULM-main/Result/ML-%j-open_3D_ulm_main_Dylan.out
#SBATCH --error=Open-3DULM-main/Result/ML-%j-open_3D_ulm_main_Dylan.err


#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paquie@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "Open-3DULM-main/scripts/open_3D_ulm_main_torch.py" --input 3D_ULM_Data --output results