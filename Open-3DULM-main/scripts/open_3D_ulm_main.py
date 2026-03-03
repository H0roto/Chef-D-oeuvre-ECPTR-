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

# Standard library
import argparse
import functools
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import cpu_count
##

# Third-party
import yaml
from loguru import logger
from tqdm import tqdm
# from tkinter import Tk, filedialog # <-- Ligne supprimée car pas d'interface graphique dans occidata
##

# Project path setup 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../src"))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
##

# Local project imports
import ulm3d.utils
import ulm3d.utils.export
import ulm3d.utils.power_doppler
import ulm3d.utils.render
import ulm3d.utils.type_config_file
from ulm3d.utils.create_archi_export import (
    create_archi_export,
    increment_config_folder,
)
from ulm3d.utils.load_data import load_iq
##
import os

def choose_workers(config: dict, backend: str) -> int:
    """
    Automatically choose the optimal number of workers depending on:
    - SLURM allocation
    - CPU vs GPU
    - YAML max_workers
    """
    yaml_max = config.get("max_workers", 1)

    # --- Detect SLURM allocation
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        available_cpus = int(slurm_cpus)
    else:
        available_cpus = os.cpu_count() or 1

    # --- GPU case (torch + CUDA)
    if backend == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                # On ne force plus à 1, mais on évite d'utiliser tous les CPU
                # 2 ou 3 workers sont souvent le point idéal ("sweet spot") pour saturer un GPU 
                # sans provoquer de Out Of Memory (OOM). 
                # Vous pouvez ajuster ce chiffre en fonction de la taille de vos cubes IQ et de votre VRAM.
                max_gpu_workers = 3 
                workers = min(yaml_max, available_cpus, max_gpu_workers)
                print(f"GPU detected → Limiting to {workers} workers to prevent VRAM overflow.")
        except Exception:
            workers = min(yaml_max, available_cpus)
    else:
        # --- Final number of workers (CPU only)
        if yaml_max == 0:
            workers = available_cpus
        else:
            workers = min(yaml_max, available_cpus)

    # --- Prevent MKL / OpenMP explosion
    if workers > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        if backend == "torch":
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass

    print(f"Using {workers} worker(s)")
    return workers

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
    "--backend",
    choices=["numpy", "torch"],
    default="numpy"
    )
    return parser.parse_args()

def configure_logger(level = "DEBUG"):
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level=level,
        format="<level>{message}</level>",
        enqueue=True
    )

def select_backend(backend_name: str):
    if backend_name == "torch":
        module = importlib.import_module("ulm3d.ulm_torch")
    else:
        module = importlib.import_module("ulm3d.ulm")

    return module

def compute_bloc(
    index: int,
    iq_files: list,
    config: dict,
    input_var_name: str,
    export_parameters: dict,
    log: list,
    ULM_class,
):   
    ulm_pipeline = ULM_class(
        iq_files=iq_files,
        **config,
        log=log,
    )
    if not logger._core.handlers:
        configure_logger(level = "DEBUG")
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

    # Tracking
    tracks = ulm_pipeline.create_tracks(localizations)
    
    if "tracking" in ulm_pipeline.log:
        print("compute_bloc sees log =", ulm_pipeline.log)
        print(f"\nINSPECTION DE LA STRUCTURE 'tracks' (Bloc {index})")
        print(f"---------------------------------------------------")
        
        # A. Vérification du conteneur principal
        print(f"1. Type de l'objet 'tracks' : {type(tracks)}")
        # On s'attend à voir : <class 'tuple'> ou <class 'list'>
        
        if isinstance(tracks, (tuple, list)):
            print(f"2. Nombre d'éléments dans 'tracks' : {len(tracks)}")
            # On s'attend à voir : 2
            
            # Vérification du premier élément (tracks[0])
            if len(tracks) > 0:
                elem_0 = tracks[0]
                print(f"3. Contenu de tracks[0] (Raw Tracks) :")
                print(f"   - Type : {type(elem_0)}")
                # On regarde la taille pour voir combien de pistes brutes il y a
                nb_raw = len(elem_0) if hasattr(elem_0, '__len__') else "Inconnu"
                print(f"   - Quantité : {nb_raw}")

            # Vérification du second élément (tracks[1])
            if len(tracks) > 1:
                elem_1 = tracks[1]
                print(f"4. Contenu de tracks[1] (Interpolated Tracks) :")
                print(f"   - Type : {type(elem_1)}")

                if hasattr(elem_1, 'shape'):
                    print(f"   - Shape (Dimensions) : {elem_1.shape}")
                else:
                    print(f"   - Taille : {len(elem_1)}")
        print(f"---------------------------------------------------\n")
        # =======================================================
    
    # Export tracks if needed.
    if "tracks" in export_parameters and tracks[1].shape[0] > 0:
        ulm3d.utils.export.export_tracks(index, tracks, export_parameters["tracks"])
    elif tracks[1].shape[0] == 0:
        logger.warning(f"No tracks detected in bloc {index}.")


def run(config_file: str, iq_files: list, output_dir: str, backend : str):
    """
    The main function of the project that runs the entire pipeline.

    Args:
        config_file (str): The path of the config file to load.
        iq_files (list): List of IQ files.
        output_dir (str): The path of the output directory.
        workers (int): Number of workers (1 for single thread).
    """
    # Load config file.
    with open(config_file) as stream:
        config = yaml.safe_load(stream)
    logger.debug(f"Input params from {config_file}:\n {yaml.dump(config)}")
    log = config.get("log", [])
    print("LOG FROM YAML =", log)
    config.pop("log", None)

    workers = choose_workers(config, backend)
    # Check if variables provided by the config file have the correct type.
    ulm3d.utils.type_config_file.check_type_config_file(config)

    config["IQ_folder_path"] = os.path.dirname(iq_files[0])

    backend_module = select_backend(backend)
    ULM_class = backend_module.ULM
    ulm_global = ULM_class(
    iq_files=iq_files,
    **config,
    log=log,
    )

    # Generate output folders configuration.
    export_parameters = create_archi_export(output_dir, config)




    input_var_name = (
        config["input_var_name"] if "input_var_name" in config else ""
    )
    # Compute Power Doppler
    if "power_doppler" in config.get("export_volume", {}):
        power_doppler = ulm3d.utils.power_doppler.compute_power_doppler(
            iq_files[0 : min(len(iq_files), 2)], ulm_global, input_var_name
        )
        ulm3d.utils.render.save_output(
            os.path.join(output_dir, "volume", "power_doppler"),
            {
                "power_doppler": power_doppler,
                "pitch": ulm_global.scale[:3],
                "origin": ulm_global.origin[:3],
            },
            export_parameters["3D_rendering"]["export_extension_volume"],
        )

    # Start 3DULM pipeline
    if workers == 1:
        logger.info(f"Start processing (single thread)")
        for ind, _ in enumerate(tqdm(iq_files)):
            compute_bloc(
                index=ind,
                iq_files=iq_files,
                config=config,
                input_var_name=input_var_name,
                export_parameters=export_parameters,
                log=log,
                ULM_class = ULM_class
            )
    else:
        logger.info(f"Start parallel pool ({workers} workers)")
        with ProcessPoolExecutor(workers) as executor:
            with tqdm(total=len(iq_files)) as pbar:
                for _ in executor.map(
                    functools.partial(
                        compute_bloc,
                        iq_files=iq_files,
                        config=config,
                        input_var_name=input_var_name,
                        export_parameters=export_parameters,
                        log=log,
                        ULM_class = ULM_class
                    ),
                    range(len(iq_files)),
                ):
                    pbar.update()

    # Export volume for visualization (always apply on interp tracks).
    ulm3d.utils.render.rendering_3d(ulm_global, export_parameters)
    logger.success(f"Processing successfully ended, save at {output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()
    # Chemin vers fichier de config
    config_file_path = args.config_file
    
    # Chemin vers dossier de données
    data_folder_path = args.input
    
    # Chemin vers dossier de résultats
    output_dir_base = args.output

    # Méthode utilisée
    backend = args.backend
    # -----------------------------------

    # Configuration du logger
    configure_logger(level = "DEBUG")

    # Vérification du fichier de config
    if not os.path.isfile(config_file_path):
        logger.error(f"Fichier de config non trouvé: {config_file_path}")
        raise FileNotFoundError("Fichier de configuration non trouvé.")
    logger.info(f"Config file: {config_file_path}")

    # Génération de la liste des fichiers IQ
    iq_files = []
    if not os.path.isdir(data_folder_path):
        raise NotADirectoryError(f"Le dossier de données n'existe pas: {data_folder_path}")

    for i in range(1, 401):
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

    # Création du dossier de sortie
    output_dir = increment_config_folder(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # Lancement du code
    run(
        config_file=config_file_path,
        iq_files=iq_files,
        output_dir=output_dir,
        backend = backend
        

    )

