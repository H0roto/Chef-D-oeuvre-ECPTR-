#! /bin/bash

#SBATCH --job-name=Transfert_mat_to_numpy
#SBATCH --output=Open-3DULM-main/Result/ML-%j-transfer_mat_to_npy_Dylan.out
#SBATCH --error=Open-3DULM-main/Result/ML-%j-transfer_mat_to_npy_Dylan.err

#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paquie@univ-tlse3.fr

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        
#SBATCH --mem=160G
#SBATCH --partition=48CPUNodes

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "Open-3DULM-main/scripts/transfert_mat_to_npy.py" --input "/projects/ecptr/3D_ULM_Data" --workers 16


