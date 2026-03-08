#! /bin/bash

#SBATCH --job-name=display_img
#SBATCH --output=Open-3DULM-main/Result/Render-%j.out
#SBATCH --error=Open-3DULM-main/Result/Render-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paquie@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      
#SBATCH --mem=40G
#SBATCH --partition=48CPUNodes

srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif \
    $HOME/my_occidata_env/bin/python -u \
    "Open-3DULM-main/scripts/display_3D_ulm.py" \
    --input "/projects/ecptr/results/config_218" \