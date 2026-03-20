import os
import argparse
import numpy as np
from scipy.io import loadmat
import mat73
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel MAT to NPY converter for 3D ULM data")
    parser.add_argument("-i", "--input", type=str, required=True, help="Folder containing .mat files")
    parser.add_argument("-v", "--var", type=str, default="IQ", help="Name of the variable to extract from MAT file")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--start", type=int, default=1, help="First file index (e.g., 1 for IQ001)")
    parser.add_argument("--end", type=int, default=400, help="Last file index (e.g., 400 for IQ400)")
    return parser.parse_args()

def convert_one_file(index, data_folder_path, input_var_name):
    """
    Reads a single .mat file (v7.3 or earlier), casts to 32-bit precision, 
    and saves as a high-performance .npy file.
    """
    mat_file = os.path.join(data_folder_path, f"IQ{index:03d}.mat")
    npy_file = os.path.join(data_folder_path, f"IQ{index:03d}.npy")
    
    if not os.path.exists(mat_file):
        return f"Skip {index}: File not found"
        
    try:
        # 1. Loading (Handle both legacy MAT and HDF5-based MAT v7.3)
        try:
            data = loadmat(mat_file, squeeze_me=True)
            iq = data[input_var_name]
        except Exception:
            # Fallback for MAT v7.3 (HDF5)
            iq = mat73.loadmat(mat_file)[input_var_name]
            
        # 2. Precision Forcing (Float32/Complex64)
        # 32-bit precision is standard for Deep Learning and drastically 
        # reduces RAM/GPU memory footprint compared to 64-bit.
        if np.iscomplexobj(iq):
            iq = iq.astype(np.complex64, copy=False)
        else:
            iq = iq.astype(np.float32, copy=False)
            
        # 3. Saving to NPY
        np.save(npy_file, iq)
        return f"Done {index}"

    except KeyError:
        return f"Error {index}: Variable '{input_var_name}' not found in file."
    except Exception as e:
        return f"Error {index}: {e}"

def main():
    args = parse_arguments()

    if not os.path.isdir(args.input):
        print(f"Error: Directory '{args.input}' does not exist.")
        return

    print(f"Starting parallel conversion on {args.workers} cores...")
    print(f"Target variable: '{args.var}' | Range: [{args.start} to {args.end}]")

    # Use ProcessPoolExecutor for true CPU parallelism (bypassing the GIL)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Fix repetitive arguments using partial
        worker_func = functools.partial(
            convert_one_file, 
            data_folder_path=args.input, 
            input_var_name=args.var
        )
        
        # Range of indices to process
        indices = range(args.start, args.end + 1)
        
        # Execute and display progress bar
        list(tqdm(executor.map(worker_func, indices), total=len(indices)))

    print("\nConversion complete! Files are now in .npy format (Float32/Complex64).")

if __name__ == "__main__":
    main()