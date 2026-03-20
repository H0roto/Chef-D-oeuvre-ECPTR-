import os
import numpy as np
from scipy import ndimage
import cv2
import scipy.io

DATA_PATH = "../../../Dataset2D/PALA_data/PALA_data_InSilicoFlow/IQ"
OUTPUT_DIR = "../../../Dataset2D/ImagesEntrainement"

DIRS = {
    "images_train": os.path.join(OUTPUT_DIR, "images/train"),
    "labels_train": os.path.join(OUTPUT_DIR, "labels/train"),
    "debug": os.path.join(OUTPUT_DIR, "debug")
}

BOX_SIZE_PIXELS = 7 

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    noise = np.random.normal(size=IQ.shape, scale=np.abs(power * impedance))
    clutter = max_IQ * 10**(clutterdB / 20)
    ampl_clutter = max_IQ * 10**((amplCullerdB + clutterdB) / 20)
    return IQ + ndimage.gaussian_filter(clutter + noise * ampl_clutter, sigma=sigmaGauss)

def meters_to_pixels(pos_m, axis_m):
    return (np.abs(axis_m - pos_m)).argmin()


def prepare_full_yolo_dataset():
    for path in DIRS.values(): os.makedirs(path, exist_ok=True)

    NoiseParam = {"power": -2, "impedance": 0.2, "sigmaGauss": 1.5, "clutterdB": -20, "amplCullerdB": 10}

    for i in range(1, 21):
        file_name = f"PALA_InSilicoFlow_IQ{i:03d}.mat"
        full_path = os.path.join(DATA_PATH, file_name)
        if not os.path.isfile(full_path): continue

        print(f"--- Traitement de {file_name} ---")
        mat_data = scipy.io.loadmat(full_path)

        iq_raw_full = np.abs(mat_data["IQ"])
        nz, nx = iq_raw_full.shape[0], iq_raw_full.shape[1]

        # --- RÉCUPÉRATION DES AXES VIA PDATA ---
        try:
            pdata = mat_data['PData'][0,0]
            # On récupère le pas (PDelta) et l'origine (Origin)
            # PDelta: [dx, dy, dz] | Origin: [x0, y0, z0]
            dx = pdata['PDelta'][0,0]
            dz = pdata['PDelta'][0,2]
            x0 = pdata['Origin'][0,0]
            z0 = pdata['Origin'][0,2]

            # Reconstruction des vecteurs d'axes
            x_axis = np.linspace(x0, x0 + (nx-1)*dx, nx)
            z_axis = np.linspace(z0, z0 + (nz-1)*dz, nz)

        except Exception as e:
            print(f"Erreur lors de l'extraction des axes dans {file_name}: {e}")
            continue

        listpos_raw = mat_data["ListPos"]
        num_frames = listpos_raw.shape[2]

        for frame_idx in range(0, num_frames, 10):
            # Traitement image
            iq_frame = iq_raw_full[:, :, frame_idx]
            noisy_iq = np.abs(PALA_AddNoiseInIQ(iq_frame, **NoiseParam))
            log_iq = 20 * np.log10(np.maximum(noisy_iq, 1e-8))
            img_png = cv2.normalize(log_iq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = img_png.shape

            base_name = f"IQ{i:03d}_frame{frame_idx:04d}"
            cv2.imwrite(os.path.join(DIRS["images_train"], f"{base_name}.png"), img_png)

            img_debug = cv2.cvtColor(img_png, cv2.COLOR_GRAY2BGR)
            label_path = os.path.join(DIRS["labels_train"], f"{base_name}.txt")

            with open(label_path, 'w') as f_out:
                for bull_idx in range(listpos_raw.shape[0]):
                    x_m = listpos_raw[bull_idx, 0, frame_idx]
                    z_m = listpos_raw[bull_idx, 2, frame_idx]

                    if np.isnan(x_m) or np.isnan(z_m): continue

                    # Conversion Mètres -> Pixels
                    x_px = meters_to_pixels(x_m, x_axis)
                    z_px = meters_to_pixels(z_m, z_axis)

                    # Normalisation YOLO
                    x_norm, y_norm = x_px / w, z_px / h
                    w_norm, h_norm = BOX_SIZE_PIXELS / w, BOX_SIZE_PIXELS / h
                    limit = BOX_SIZE_PIXELS//2
                    if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                        f_out.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        # Debug visuel
                        cv2.rectangle(img_debug, (x_px-limit, z_px-limit), (x_px+limit, z_px+limit), (0, 255, 0), 1)

            cv2.imwrite(os.path.join(DIRS["debug"], f"{base_name}_debug.png"), img_debug)

if __name__ == "__main__":
    prepare_full_yolo_dataset()
    print("Terminé !")