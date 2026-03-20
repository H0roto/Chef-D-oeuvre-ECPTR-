"""
This is a displaying script of Open_3DULM.
ULM volumes are load and displayed.
"""

import argparse
import os
import sys
# from tkinter import Tk, filedialog
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../src")) 

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ulm3d.utils.load_data import load_volume

mpl.rcParams["xtick.labelsize"] = "x-small"
mpl.rcParams["ytick.labelsize"] = "x-small"


def parse_arguments():
    parser = argparse.ArgumentParser(description="3D ULM reconstruction")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input volume dir (all files we be selected) (optional, else dialogbox)",
    )
    parser.add_argument(
        "-v",
        "--verbose-level",
        type=int,
        default=1,
        choices=range(4),
        help="Verbosity level (0: warning, 1: info, 2: debug, 3: trace)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="mm",
        choices=("pixel", "mm"),
        help="Scale pixel",
    )
    parser.add_argument("--show", action="store_true", help="Show images")
    return parser.parse_args()


def get_data(file: str):
    content = load_volume(file)
    mat = None
    label = ""
    origin = np.zeros(3)
    for key, val in content.items():
        if isinstance(val, np.ndarray):
            if val.size > 5:
                mat = val
                label = key
                break

    if "pitch" in content:
        pitch = content["pitch"].squeeze()
    else:
        pitch = np.ones(mat.ndim)
        logger.warning(f"Missing pitch if {file}")
    if "origin" in content:
        origin = content["origin"].squeeze()

    if mat is None:
        logger.warning(f"No volume found in {file}")
    else:
        logger.info(f"Volume {label} ({mat.shape}, pitch {pitch}) found in {file}")
    return mat, label, pitch, origin


def export_rendering(vol_files: list, show: bool, scale: bool):

    for i, file in enumerate(vol_files):
        logger.info(f"Render ({i+1}/{len(vol_files)}) file {file}")
        mat, label, pitch, origin = get_data(file)
        if mat is None:
            continue
        if mat.ndim == 4:
            mat = np.sum(mat, -1)
        if mat.ndim != 3:
            continue

        func = np.mean
        compress_power = 1
        cmap = "hot"
        mat_proj = np.power(func(mat, axis=1), compress_power)
        clim = np.array([np.nanmin(mat_proj), np.nanmax(mat_proj)])

        norm = mcolors.Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        if "density" in file:
            compress_power = 0.8
            clim = np.power(clim, compress_power)
        elif "doppler" in file:
            compress_power = 3 / 4
            clim = np.power(clim, compress_power)
        elif "directivity" in file:
            compress_power = 0.8
            cmap = "twilight"
            clim = np.array([-1, 1]) * np.max(np.abs(clim)) * 0.5
        elif "velocity" in file:
            cmap = "jet"
            func = np.max
            clim = [0, func(mat)]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"{label} (pitch: {pitch}, compression {compress_power})", size="medium"
        )
        for ind, ax_name in enumerate(list("ZYX")):
            mat_proj = func(mat, axis=ind)
            pitch_proj = np.delete(pitch, ind)
            ax_proj = np.delete(list("ZYX"), ind)
            origin_proj = np.delete(origin, ind)

            extent = -0.5 + np.array([0, mat_proj.shape[1], mat_proj.shape[0], 0])
            if scale:
                extent[:2] = extent[:2] * pitch_proj[1] + origin_proj[1]
                extent[2:] = extent[2:] * pitch_proj[0] + origin_proj[0]

            im = axs[ind].imshow(
                np.sign(mat_proj) * np.power(np.abs(mat_proj), compress_power),
                aspect="equal",
                cmap=cmap,
                clim=clim,
                extent=extent,
            )
            axs[ind].set_title(f"MIP Axe {ax_name}")
            if scale:
                axs[ind].set(xlabel=f"{ax_proj[1]} [mm]", ylabel=f"{ax_proj[0]} mm")

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="horizontal",
        )
        fig_path = os.path.join(os.path.dirname(file), label + ".png")
        logger.info(f"Figure saved at {fig_path}")
        fig.savefig(fig_path)

        if show:
            plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    args = parse_arguments()

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=["WARNING", "INFO", "DEBUG", "TRACE"][args.verbose_level],
        format="[<green>{time:HH:mm:ss}</green> <d>({elapsed.seconds}s)</d>] <level>{level: <5}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    if args.input is None:
        logger.error("Aucun dossier d'entrée spécifié. Utilisez --input.")
        sys.exit(1)
        
    input_path = os.path.abspath(args.input)
    volume_files = []
    
    logger.info(f"Recherche des volumes dans : {input_path}")
    for root, dirs, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith((".hdf5", ".npz")):
                volume_files.append(os.path.join(root, f))

    if not volume_files:
        logger.warning(f"Aucun fichier HDF5/NPZ trouvé dans {input_path} ou ses sous-dossiers.")
        sys.exit(0)

    logger.success(f"{len(volume_files)} fichiers de volume trouvés au total !")

    export_rendering(
        vol_files=volume_files,
        show=args.show,
        scale=(args.scale == "mm"),
    )
