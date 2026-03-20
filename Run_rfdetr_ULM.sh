#! /bin/bash

#SBATCH --job-name=Test_yolo_radial_sym
#SBATCH --output=Open-3DULM-main/Result/RFDETR/ML-%j-open_3D_ulm_rfdetr.out
#SBATCH --error=Open-3DULM-main/Result/RFDETR/ML-%j-open_3D_ulm_rfdetr.err


#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.emile@univ-tlse3.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX6000Node
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
# #SBATCH --mem=80G
# srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "Open-3DULM-main/scripts/open_3D_ulm_main.py" --input "/projects/ecptr/3D_ULM_Data" --output "/projects/ecptr/results/" --backend "yolo" --config-file "/projects/ecptr/Espace_GIT/Espace_Dylan/Chef-D-oeuvre-ECPTR-/Open-3DULM-main/config/basic_config.yaml" --yolo-model "YOLO/Entrainement_YOLO12s_hyper_param_15dB_upgrade/weights/best.pt" 
srun singularity exec /apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif $HOME/my_occidata_env/bin/python -u "Open-3DULM-main/scripts/open_3D_ulm_main.py" --input "/projects/ecptr/3D_ULM_Data" --output "/projects/ecptr/results/rfdetr" --backend "rfdetr" --config-file "/projects/ecptr/Espace_GIT/Espace_Enzo/Chef-D-oeuvre-ECPTR-/Open-3DULM-main/config/basic_config.yaml" --rfdetr-model "RFDETR/Entrainement_RFDETR/checkpoint.pth"