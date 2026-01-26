"""
This is the main script of Open_3DULM.
Ultrasound data acquisition and beamforming have already been done before this part.

Input:
    User has to provide:
        - The raw beamformed data (IQ: 3D + time)
        - YAML path of the config file containing input data and export (I/O section), ULM, acquisition and image reconstruction (beamforming) parameters.

Output:
    This script will create an export folder named config_{index} according to the path provided by the user and export microbubble localizations, tracks (raw and interpolaed),
    and 3D density and velocity rendering as specified in the config file.

Details:
    - The YAML config file is loaded.
    - IQ data paths are collected.
    - The ULM class is created and used for the entire pipeline.
    - If "max_workers" > 1 in the config file, the code runs in parallel on the CPU.
    - Each IQ dataset will be processed as follows:
        - IQ filtering (SVD and/or Bandpass filtering).
        - Sub-wavelegnth localization of microbubbles using the 3D radial symmetry (and saving if needed).
        - Tracking of microbubbles (and saving if needed).
    - If 3D rendering export is needed, the code will collect all generated tracks and create volumes for visualization purposes (Density and/or Velocity).
"""

import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
# from tkinter import Tk, filedialog # <-- Ligne supprimée
import sys

import yaml
from loguru import logger
from tqdm import tqdm

import ulm3d.ulm
import ulm3d.utils
import ulm3d.utils.export
import ulm3d.utils.power_doppler
import ulm3d.utils.render
import ulm3d.utils.type_config_file
from ulm3d.utils.create_archi_export import (create_archi_export,
                                             increment_config_folder)
from ulm3d.utils.load_data import load_iq


def parse_arguments():
    parser = argparse.ArgumentParser(description="3D ULM reconstruction")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Config file path (optional, else dialogbox)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input IQ dir (all files we be selected) (optional, else dialogbox)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output dir (optional, else dialogbox)",
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
        "--workers",
        type=int,
        default=None,
        help="Parallel computing (use value in config.yaml by default)",
    )
    return parser.parse_args()


def compute_bloc(
    ulm_pipeline: ulm3d.ulm.ULM,
    iq_files: list,
    input_var_name: str,
    export_parameters: dict,
    index: int,
):
    """
    Function to apply 3DULM for each 3D ultrasound data block (it can be done in parallel).

    Args:
        ulm_pipeline (ULM): The ULM structure to execute the 3DULM pipeline.
        iq_files (str): The path of the IQ file to load.
        input_var_name (str): The name of the input variable in the dictionary .mat to open the IQ file.
        export_parameters (dict): The dictionary that contains settings to export data.
        index (int): The index of the current block to apply 3D ULM.
    """
    # Load IQ.
    iq = load_iq(iq_files[index], input_var_name)
    # Filtering.
    if ulm_pipeline.filt_mode != "no_filter":
        iq_before_loc = ulm_pipeline.filtering(iq)  # Filtering is applied.
    else:
        iq_before_loc = iq  # IQ is used without filtering.
    # Super-localization.
    localizations = ulm_pipeline.super_localization(iq_before_loc)
    # Export localizations if needed.
    if "localizations" in export_parameters and localizations.shape[0] > 0:
        ulm3d.utils.export.export_locs(
            index, localizations, export_parameters["localizations"]
        )
    elif localizations.shape[0] == 0:
        logger.warning(f"No localizations detected in bloc {index}.")

    # Tracking.
    tracks = ulm_pipeline.create_tracks(localizations)

    # Export tracks if needed.
    if "tracks" in export_parameters and tracks[1].shape[0] > 0:
        ulm3d.utils.export.export_tracks(index, tracks, export_parameters["tracks"])
    elif tracks[1].shape[0] == 0:
        logger.warning(f"No tracks detected in bloc {index}.")


def run(config_file: str, iq_files: list, output_dir: str, workers: int):
    """
    The main function of the project that runs the entire pipeline.

    Args:
        config_file (str): The path of the config file to load.
        iq_files (list): List of IQ files.
        output_dir (str): The path of the output directory.
        workers (int): Number of workers (1 for single thread).
    """
    Workers = 0 
    if workers == 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") \
                     or os.environ.get("SLURM_JOB_CPUS_PER_NODE")

        if slurm_cpus:
            workers = int(slurm_cpus)
        else:
            workers = cpu_count()

        logger.info(f"Auto workers = {workers}")
    # Load config file.
    with open(config_file) as stream:
        config = yaml.safe_load(stream)
    logger.debug(f"Input params from {config_file}:\n {yaml.dump(config)}")

    if workers is None:
        workers = config["max_workers"]
        logger.info(f"Use default number of workers {workers}")

    # Check if variables provided by the config file have the correct type.
    ulm3d.utils.type_config_file.check_type_config_file(config)

    config["IQ_folder_path"] = os.path.dirname(iq_files[0])

    # Generate output folders configuration.
    export_parameters = create_archi_export(output_dir, config)

    # Create 3DULM class.
    ulm = ulm3d.ulm.ULM(iq_files=iq_files, **config)

    input_var_name = input_var_name = (
        config["input_var_name"] if "input_var_name" in config else ""
    )
    # Compute Power Doppler
    if "power_doppler" in config["export_volume"]:
        power_doppler = ulm3d.utils.power_doppler.compute_power_doppler(
            iq_files[0 : min(len(iq_files), 2)], ulm, input_var_name
        )
        ulm3d.utils.render.save_output(
            os.path.join(output_dir, "volume", "power_doppler"),
            {
                "power_doppler": power_doppler,
                "pitch": ulm.scale[:3],
                "origin": ulm.origin[:3],
            },
            export_parameters["3D_rendering"]["export_extension_volume"],
        )

    # Start 3DULM pipeline
    if workers == 1:
        logger.info(f"Start processing (single thread)")
        for ind, _ in enumerate(tqdm(iq_files)):
            compute_bloc(
                ulm,
                iq_files,
                input_var_name,
                export_parameters,
                ind,
            )
    else:
        workers = min(workers, cpu_count())
        logger.info(f"Start parallel pool ({workers} workers)")
        with ProcessPoolExecutor(workers) as executor:
            with tqdm(total=len(iq_files)) as pbar:
                for _ in executor.map(
                    functools.partial(
                        compute_bloc,
                        ulm,
                        iq_files,
                        input_var_name,
                        export_parameters,
                    ),
                    range(len(iq_files)),
                ):
                    pbar.update()

    # Export volume for visualization (always apply on interp tracks).
    if "3D_rendering" in export_parameters:
        ulm3d.utils.render.rendering_3d(ulm, export_parameters)
    logger.success(f"Processing successfully ended, save at {output_dir}")


if __name__ == "__main__":

    # --- 1. VOS CHEMINS SONT DEFINIS ICI ---
    # Chemin vers votre fichier de config
    config_file_path = "/projects/ecptr/Open-3DULM-main-test/config/basic_config.yaml"
    
    # Chemin vers votre DOSSIER de données (celui avec les 400 .mat)
    data_folder_path = "/projects/ecptr/3D_ULM_Data" 
    
    # Chemin vers votre DOSSIER de résultats
    output_dir_base = "/projects/ecptr/results/"
    # -----------------------------------

    # Configuration du logger (ne pas changer)
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO", # Niveau de verbosité
        format="[<green>{time:HH:mm:ss}</green> <d>({elapsed.seconds}s)</d>] <level>{level: <5}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    logger.info("CHEMINS DEFINIS EN DUR DANS LE SCRIPT")

    # Vérification du fichier de config
    if not os.path.isfile(config_file_path):
        logger.error(f"Fichier de config non trouvé: {config_file_path}")
        raise FileNotFoundError("Fichier de configuration non trouvé.")
    logger.info(f"Config file: {config_file_path}")

    # --- Génération de la liste des 400 fichiers IQ ---
    iq_files = []
    if not os.path.isdir(data_folder_path):
        raise NotADirectoryError(f"Le dossier de données n'existe pas: {data_folder_path}")

    for i in range(1, 401):  # Crée une boucle de 1 à 400
        # Formate le nom du fichier : "IQ" + numéro (ex: 001) + ".mat"
        file_name = f"IQ{i:03d}.mat"
        
        # Crée le chemin complet vers le fichier
        full_path = os.path.join(data_folder_path, file_name)
        
        # Ajoute le chemin complet à la liste
        iq_files.append(full_path)

    # Vérification que le premier fichier existe
    if not os.path.isfile(iq_files[0]):
        logger.warning(f"ATTENTION: Le premier fichier {iq_files[0]} n'a pas été trouvé.")
        logger.warning("Vérifiez que 'data_folder_path' est correct et que vos fichiers sont bien nommés (IQ001.mat, ...)")
    
    logger.info(
        f"{len(iq_files)} IQ files générés: {data_folder_path} ({iq_files[0]}...)"
    )
    # --- Fin de la modification ---

    # Création du dossier de sortie
    output_dir = increment_config_folder(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # Lancement du code
    run(
        config_file=config_file_path,
        iq_files=iq_files,
        output_dir=output_dir,
        workers=None, # Utilise la valeur de 'max_workers' du fichier config
    )