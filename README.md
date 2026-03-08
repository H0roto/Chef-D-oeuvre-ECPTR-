# Open-3DULM : 3D Ultrasound Localization Microscopy Pipeline

![YOLO Rendering](Open-3DULM-main/doc/yolo.png)

This repository contains the complete source code for the 3D **ULM (Ultrasound Localization Microscopy)** pipeline, designed for microbubble detection and tracking. 

To address various performance and experimental requirements, the code has been split into **3 distinct approaches**:
1. **CPU Approach (NumPy)**: The baseline method, without hardware-specific optimization.
2. **GPU Approach (PyTorch)**: Massive optimization utilizing tensor operations on the graphics card.
3. **YOLO Approach (GPU + AI)**: Replaces the classic Region of Interest (ROI) extraction with a convolutional neural network.

---

## 🛠️ Installation & Setup

To install the pipeline and all its dependencies (including YOLO and tracking tools) in your environment:

1. **Activate your environment**:
   ```bash
   source $HOME/my_occidata_env/bin/activate
   ```

2. **Install in editable mode**:
   ```bash
   cd Open-3DULM-main
   pip install -e .
   ```
*This command uses `setup.py` to automatically install requirements and links the `ulm3d` package to your source folder for immediate updates.*

---

## 📁 Repository Contents & Code Architecture

### 🎬 Main Launcher
* **`open_3D_ulm_main.py`**: This is the central script (the entry point). It reads the YAML configuration file, handles multiprocessing (if applicable), and instantiates the correct ULM class based on the chosen backend (`numpy`, `torch`, or `yolo`).

### ⚙️ Core ULM Pipeline (Classes)
These files contain the overall logic of the ULM method (filtering, localization, tracking):
* **`ulm.py`**: Default (legacy) file. Executes the classic pipeline on the CPU by calling the NumPy version of the radial symmetry function.
* **`ulm_torch.py`**: GPU-optimized version. The structure is similar to `ulm.py`, but it leverages PyTorch to drastically accelerate computations and calls the optimized radial symmetry function.
* **`ulm_yolo.py`**: Hybrid AI/Algorithmic version. It retains PyTorch's GPU optimization but replaces the classic local maxima detection method (ROI calculation) with 3D inference via **YOLO** on MIP (Maximum Intensity Projection) projections.

### 🧮 Localization Algorithms (Sub-pixel)
These scripts handle super-resolved localization at the geometric center of the microbubbles:
* **`radial_symmetry_center_numpy.py`**: 3D radial symmetry localization algorithm implemented in a standard way (loops and CPU calculations).
* **`radial_symmetry_center_torch.py`**: The same mathematical algorithm, rewritten for batched processing on the GPU via PyTorch. It includes a robust linear system resolution (Tikhonov Regularization).

### 🔄 Data Conversion (MAT to NPY)
The pipeline is optimized for `.npy` files to ensure maximum I/O speed. If your data is in MATLAB `.mat` format, use the provided conversion script:

```bash
python convert_mat_to_npy.py --input "path/to/data" --var "IQ"
```
*or launch the `Run_transfer_mat_to_npy.sh` file with the correct parameters.*

### 🖼️ Visualization & Rendering
* **`display_3D_ulm.py`**: A post-processing script that recursively scans result directories for generated volumes (`.hdf5` or `.npz`) and automatically renders 2D Maximum Intensity Projections (MIPs) for Density, Velocity, etc. It runs in headless mode (no GUI) ensuring compatibility with cluster environments.

### 🤖 YOLO Utilities (Artificial Intelligence)
These supplementary scripts are only necessary if you want to retrain the AI detection model:
* **`prepare_yolo_dataset.py`**: Data generation script. It takes raw IQ data, injects customizable realistic noise, extracts frames, and creates labels (bounding boxes) in the standard YOLO format for `train` and `val` sets.
* **`training_YOLO.py`**: Training script (Fine-tuning). It loads a pre-trained model (e.g., YOLO12s) and runs the learning process on the generated dataset to adapt it specifically to our microbubble signatures.

---

# 🚀 Open-3DULM: SLURM Execution Scripts

This repository contains bash scripts (`.sh`) to run the Open-3DULM pipeline on a computing cluster using **SLURM**. 

The pipeline has been developed in several versions to exploit different computing methods for Region of Interest (ROI) localization: **NumPy (CPU)**, **PyTorch (GPU)**, and **YOLO (GPU)**. 

All scripts use a Singularity container (`pytorch-NGC-25-01.sif`) ensuring a stable execution environment pre-installed with CUDA 12 and PyTorch.

---

## 🔬 1. ULM Pipeline Execution (Inference & Reconstruction)

These three scripts execute the complete 3D ULM reconstruction pipeline (`open_3D_ulm_main.py`). The choice of script depends on the desired backend (compute engine).

### 🧠 `Run_yolo_ULM.sh` (Backend: YOLO on GPU)
This script runs the pipeline using an Artificial Intelligence model (**YOLO**) to intelligently detect microbubble Regions of Interest (ROI) via MIP (Maximum Intensity Projection) projections.

* **Partition**: `RTX8000Nodes`
* **Allocated Resources**: 1 GPU, 4 CPUs (`--cpus-per-task=4`)
* **Specifics**:
  * The `--backend "yolo"` parameter activates inference via the YOLO architecture.
  * Requires the additional `--yolo-model` parameter to point to the previously trained model weights (e.g., `best.pt`).
  * The 4 CPUs are allocated to ensure a fast data pipeline (preprocessing and block loading) to the GPU, preventing data-loading bottlenecks.

### ⚡ `Run_torch_ULM.sh` (Backend: PyTorch on GPU)
This script uses the classic analytical method for local maxima detection but massively parallelizes it on the graphics card using **PyTorch** tensors.

* **Partition**: `RTX8000Nodes`
* **Allocated Resources**: 1 GPU, 2 CPUs (`--cpus-per-task=2`)
* **Specifics**:
  * The `--backend "torch"` parameter enables native hardware acceleration.
  * Requires fewer CPUs than the YOLO version because heavy mathematical operations (3D max pooling, spatial convolutions) are handled directly by the GPU. Execution is extremely fast and highly optimized.

### 🐢 `Run_numpy_ULM.sh` (Backend: NumPy on CPU)
This script runs the historical algorithmic pipeline entirely on the processor (CPU). It serves primarily as a baseline or as a fallback solution if GPU nodes are unavailable.

* **Partition**: `48CPUNodes`
* **Allocated Resources**: 48 CPUs (`--cpus-per-task=48`), **160 GB RAM** (`--mem=160G`)
* **Specifics**:
  * The `--backend "numpy"` parameter forces CPU computation.
  * **Massive Memory**: 3D matrix calculations on the CPU require a huge amount of RAM to store raw data blocks without the memory optimization of a GPU.
  * Processing is distributed across 48 logical cores to compensate for the CPU's slower calculation speed compared to the thousands of cores on a GPU.

---

## 🎯 2. Preparation and Training Scripts (YOLO)

For the YOLO backend to function during inference, a model must first be trained on our in silico flow data. These scripts manage the creation of the dataset and the learning process.

### 🛠️ `Run_prepare_YOLO_dataset.sh`
This script automatically generates the dataset for YOLO training from raw IQ data (`prepare_yolo_dataset.py`).

* **Allocated Resources**: 1 GPU, 1 CPU.
* **How it works**:
  * **Artificial Noise**: Generates images with various noise levels defined by the `--noise-levels 10 15 20` parameter (in dB).
  * **Architecture**: Automatically creates the YOLO folder structure (`images/` and `labels/` separated into `train` and `val`) and generates portable `dataset.yaml` configuration files.
  * **Formatting**: Sets the bounding box size around the microbubbles via `--box-size 5`.

### 🏋️ `Run_train_YOLO.sh`
This script launches the actual training of the artificial intelligence detection model (`training_YOLO.py`).

* **Allocated Resources**: 1 GPU, 1 CPU.
* **How it works**:
  * Uses a base pre-trained model defined by `--model "YOLO/yolo12s.pt"`.
  * Fine-tunes the neural network on a specific dataset targeted by `--data "YOLO/dataset_yolo_15dB/dataset.yaml"`.
  * Exports the results (training logs, learning curves, and final weights `best.pt`) to the folder specified by `--project` and `--name`.

---

## 📊 3. Post-Processing & Visualization

Once the ULM pipeline has generated the output volumes, you can automatically render the 2D projection images (MIPs) directly on the cluster without needing to download the heavy raw arrays to your local machine.

### 🖼️ `Run_display_img.sh`
This script recursively searches for `.hdf5` and `.npz` volumes in your results folder and generates `.png` images (Density, Velocity, Power Doppler) in their respective directories.

* **Allocated Resources**: 1 CPU, 40 GB RAM (Image rendering requires significantly less memory and compute power than the ULM pipeline).
* **How it works**:
  * Runs `display_3D_ulm.py` in headless mode (`matplotlib.use('Agg')`) to prevent display/GUI errors on the Occidata cluster.
  * You simply provide the root results directory via the `--input` argument, and the script handles the recursive search and rendering automatically.