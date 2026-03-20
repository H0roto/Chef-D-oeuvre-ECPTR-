from rfdetr import RFDETRSmall

import os, torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

model = RFDETRSmall()

model.train(
    dataset_dir="projects/ecptr/Espace_GIT/Espace_Dylan/Chef-D-oeuvre-ECPTR-/YOLO/dataset_yolo_15dB",
    epochs=80,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    use_ema = True,
    output_dir="/projects/ecptr/Espace_GIT/Espace_Thomas/results/rfdetr/run4"
)




