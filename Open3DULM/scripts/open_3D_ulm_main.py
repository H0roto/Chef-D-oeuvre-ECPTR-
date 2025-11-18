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
from tkinter import Tk, filedialog

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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking of pipeline steps",
    )
    return parser.parse_args()


def compute_bloc(
    ulm_pipeline: ulm3d.ulm.ULM,
    iq_files: list,
    input_var_name: str,
    export_parameters: dict,
    index: int,
    enable_benchmark: bool = False,
):
    """
    Function to apply 3DULM for each 3D ultrasound data block (it can be done in parallel).

    Args:
        ulm_pipeline (ULM): The ULM structure to execute the 3DULM pipeline.
        iq_files (str): The path of the IQ file to load.
        input_var_name (str): The name of the input variable in the dictionary .mat to open the IQ file.
        export_parameters (dict): The dictionary that contains settings to export data.
        index (int): The index of the current block to apply 3D ULM.
        enable_benchmark (bool): Enable benchmarking for this worker.
    
    Returns:
        dict: Benchmark timings if enabled, None otherwise.
    """
    import time
    timings = {} if enable_benchmark else None
    
    # Load IQ.
    iq = load_iq(iq_files[index], input_var_name)

    # Filtering.
    if ulm_pipeline.filt_mode != "no_filter":
        if enable_benchmark:
            start = time.perf_counter()
        iq_before_loc = ulm_pipeline.filtering(iq)  # Filtering is applied.
        if enable_benchmark:
            timings['filtering'] = time.perf_counter() - start
    else:
        iq_before_loc = iq  # IQ is used without filtering.

    # Super-localization.
    if enable_benchmark:
        start = time.perf_counter()
    localizations = ulm_pipeline.super_localization(iq_before_loc)
    if enable_benchmark:
        timings['super_localization'] = time.perf_counter() - start

    # Export localizations if needed.
    if "localizations" in export_parameters and localizations.shape[0] > 0:
        ulm3d.utils.export.export_locs(
            index, localizations, export_parameters["localizations"]
        )
    elif localizations.shape[0] == 0:
        logger.warning(f"No localizations detected in bloc {index}.")

    # Tracking.
    if enable_benchmark:
        start = time.perf_counter()
    tracks = ulm_pipeline.create_tracks(localizations)
    if enable_benchmark:
        timings['tracking'] = time.perf_counter() - start

    # Export tracks if needed.
    if "tracks" in export_parameters and tracks[1].shape[0] > 0:
        ulm3d.utils.export.export_tracks(index, tracks, export_parameters["tracks"])
    elif tracks[1].shape[0] == 0:
        logger.warning(f"No tracks detected in bloc {index}.")
    
    return timings


def run(config_file: str, iq_files: list, output_dir: str, workers: int, enable_benchmark: bool = False):
    """
    The main function of the project that runs the entire pipeline.

    Args:
        config_file (str): The path of the config file to load.
        iq_files (list): List of IQ files.
        output_dir (str): The path of the output directory.
        workers (int): Number of workers (1 for single thread).
        enable_benchmark (bool): Enable benchmarking of pipeline steps.
    """

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
    
    # Initialize benchmarking if enabled
    benchmark_manager = None
    if enable_benchmark:
        from ulm3d.utils.benchmark import BenchmarkManager
        benchmark_manager = BenchmarkManager(output_dir)
        benchmark_manager.start_total()
        ulm.benchmark_manager = benchmark_manager
        logger.info("Benchmarking enabled")

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
                enable_benchmark,
            )
    else:
        workers = min(workers, cpu_count())
        logger.info(f"Start parallel pool ({workers} workers)")
        with ProcessPoolExecutor(workers) as executor:
            with tqdm(total=len(iq_files)) as pbar:
                results = []
                for result in executor.map(
                    functools.partial(
                        compute_bloc,
                        ulm,
                        iq_files,
                        input_var_name,
                        export_parameters,
                        enable_benchmark=enable_benchmark,
                    ),
                    range(len(iq_files)),
                ):
                    if enable_benchmark and result:
                        results.append(result)
                    pbar.update()
        
        # Aggregate benchmark results from parallel workers
        if enable_benchmark and results and benchmark_manager:
            for timings in results:
                for step_name, duration in timings.items():
                    benchmark_manager.record(step_name, duration)

    # Export volume for visualization (always apply on interp tracks).
    if "3D_rendering" in export_parameters:
        ulm3d.utils.render.rendering_3d(ulm, export_parameters)
    
    # Export benchmark results if enabled
    if enable_benchmark and benchmark_manager:
        benchmark_manager.stop_total()
        benchmark_manager.print_summary()
        benchmark_manager.export_to_csv()
    
    logger.success(f"Processing successfully ended, save at {output_dir}")


if __name__ == "__main__":

    # Get path of the config file with visual interface.
    args = parse_arguments()

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level=["WARNING", "INFO", "DEBUG", "TRACE"][args.verbose_level],
        format="[<green>{time:HH:mm:ss}</green> <d>({elapsed.seconds}s)</d>] <level>{level: <5}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # Set input config file
    if args.config_file is None:
        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        config_file_path = filedialog.askopenfilename(
            initialdir="config/",
            title="Select config file",
            filetypes=(("yaml files", "*.yaml"), ("all files", "*.*")),
            parent=root,
        )
    else:
        config_file_path = os.path.abspath(args.config_file)

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(
            "Please be sure to select a config file with the dialogbox."
        )
    logger.info(f"Config file: {config_file_path}")
    # Set input IQ files
    iq_files = []
    if args.input is None:
        # Get IQ folder with visual interface.
        root = Tk()
        root.withdraw()
        iq_files = filedialog.askopenfilenames(
            initialdir=".",
            title="Select IQ files",
            filetypes=(
                ("mat files", "*.mat"),
                ("npz files", "*.npz"),
                ("npy files", "*.npy"),
                ("all files", "*.*"),
            ),
        )
    else:
        for file in os.listdir(os.path.abspath(args.input)):
            if file.endswith(".mat"):
                iq_files.append(os.path.join(os.path.abspath(args.input), file))

    if len(iq_files) == 0:
        logger.error("No files selected")
        raise FileNotFoundError("Please be sure to select IQ files to import data.")
    logger.info(
        f"{len(iq_files)} IQ files: {os.path.dirname(iq_files[0])} ({iq_files[0]}...)"
    )

    # Set output dir
    if args.output is None:
        # Get output folder with visual interface.
        output_dir = filedialog.askdirectory(
            initialdir=".",
            title="Select output folder",
        )
    else:
        output_dir = os.path.abspath(args.output)
    if output_dir == "":
        raise FileNotFoundError(
            "Please be sure to select an output folder to export data."
        )

    output_dir = increment_config_folder(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    run(
        config_file=config_file_path,
        iq_files=iq_files,
        output_dir=output_dir,
        workers=args.workers,
        enable_benchmark=args.benchmark,
    )
