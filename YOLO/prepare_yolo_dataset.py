import os
import numpy as np
from scipy import ndimage
import cv2
import scipy.io
import argparse

np.random.seed(42)

# ==========================================
# 1. CONFIGURATION
# ==========================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="3D ULM reconstruction")
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="PALA_data_InSilicoFlow/IQ", 
        help="Input IQ directory"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="dataset_yolo", 
        help="Base output directory"
    )
    parser.add_argument(
        "--box-size", 
        type=int, 
        default=5, 
        help="Bounding box size in pixels"
    )
    parser.add_argument(
        "--noise-levels", 
        nargs="+", 
        type=int, 
        default=[10,15,20], 
        help="List of noise levels (amplCullerdB)"
    )
    return parser.parse_args()

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    noise = np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape)
    clutter_term = max_IQ * 10**(clutterdB / 20)
    noise_term = noise * max_IQ * 10**((amplCullerdB + clutterdB) / 20)
    return IQ + ndimage.gaussian_filter(clutter_term + noise_term, sigma=sigmaGauss)

def meters_to_pixels(pos_m, axis_m):
    return (np.abs(axis_m - pos_m)).argmin()

# ==========================================
# 2. GENERATION (WITH TRAIN/VAL SPLIT)
# ==========================================

def prepare_full_yolo_dataset(args):
    input_path = args.input
    base_output_dir = args.output
    box_size = args.box_size
    noise_levels = args.noise_levels

    for db_val in noise_levels:
        print(f"\nCREATING DATASET FOR amplCullerdB = {db_val} dB")
        current_output_dir = f"{base_output_dir}_{db_val}dB"
        
        dirs = {
            "images_train": os.path.join(current_output_dir, "images/train"),
            "labels_train": os.path.join(current_output_dir, "labels/train"),
            "images_val": os.path.join(current_output_dir, "images/val"),
            "labels_val": os.path.join(current_output_dir, "labels/val"),
        }
        for path in dirs.values(): 
            os.makedirs(path, exist_ok=True)
        yaml_path = os.path.join(current_output_dir, "dataset.yaml")
        abs_dataset_dir = os.path.abspath(current_output_dir)
        with open(yaml_path, 'w') as f_yaml:
            f_yaml.write(f"path: {abs_dataset_dir}\n") # Chemin absolu ici
            f_yaml.write(f"train: images/train\n")
            f_yaml.write(f"val: images/val\n\n")
            f_yaml.write(f"nc: 1\n")
            f_yaml.write(f"names: ['microbubble']\n")

        noise_params = {
            "power": -2, 
            "impedance": 0.2, 
            "sigmaGauss": 1.5, 
            "clutterdB": -20, 
            "amplCullerdB": db_val
        }
        
        for i in range(1, 21):
            file_name = f"PALA_InSilicoFlow_IQ{i:03d}.mat"
            full_path = os.path.join(input_path, file_name)
            if not os.path.isfile(full_path): 
                continue
            
            # Train/Val split logic
            target_dir = "train" if i <= 16 else "val"
                
            print(f"--- File {i:02d}/20 -> Moving to [{target_dir.upper()}] folder ---")
            
            mat_data = scipy.io.loadmat(full_path)
            iq_raw_full = np.abs(mat_data["IQ"])
            nz, nx = iq_raw_full.shape[0], iq_raw_full.shape[1]

            try:
                pdata = mat_data['PData'][0,0]
                dx, dz = pdata['PDelta'][0,0], pdata['PDelta'][0,2]
                x0, z0 = pdata['Origin'][0,0], pdata['Origin'][0,2]
                x_axis = np.linspace(x0, x0 + (nx-1)*dx, nx)
                z_axis = np.linspace(z0, z0 + (nz-1)*dz, nz)
            except Exception:
                continue

            listpos_raw = mat_data["ListPos"]
            num_frames = listpos_raw.shape[2]

            for frame_idx in range(0, num_frames, 10):
                iq_frame = iq_raw_full[:, :, frame_idx]
                noisy_iq = np.abs(PALA_AddNoiseInIQ(iq_frame, **noise_params))
                img_png = cv2.normalize(noisy_iq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h, w = img_png.shape
                
                base_name = f"IQ{i:03d}_frame{frame_idx:04d}"
                
                img_path = os.path.join(dirs[f"images_{target_dir}"], f"{base_name}.png")
                lbl_path = os.path.join(dirs[f"labels_{target_dir}"], f"{base_name}.txt")
                
                cv2.imwrite(img_path, img_png)
                
                with open(lbl_path, 'w') as f_out:
                    for bull_idx in range(listpos_raw.shape[0]):
                        x_m = listpos_raw[bull_idx, 0, frame_idx]
                        z_m = listpos_raw[bull_idx, 2, frame_idx]
                        
                        if np.isnan(x_m) or np.isnan(z_m): 
                            continue

                        x_px = meters_to_pixels(x_m, x_axis)
                        z_px = meters_to_pixels(z_m, z_axis)

                        x_norm, y_norm = x_px / w, z_px / h
                        w_norm, h_norm = box_size / w, box_size / h
                        
                        if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                            f_out.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == "__main__":
    args = parse_arguments()
    prepare_full_yolo_dataset(args)
    print("Train/Val datasets generation completed!")