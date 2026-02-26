import os
import numpy as np
from scipy import ndimage
import cv2
import scipy.io

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = "/projects/ecptr/Dataset2D"
OUTPUT_DIR = "/projects/ecptr/Espace_GIT/Espace_Dylan/Chef-D-oeuvre-ECPTR-/resultat_img"

DIRS = {
    "images_train": os.path.join(OUTPUT_DIR, "images/train"),
    "labels_train": os.path.join(OUTPUT_DIR, "labels/train"),
}

BOX_SIZE_PIXELS = 5 # Taille de la boite en pixels pour l'affichage/YOLO

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    noise = np.random.normal(size=IQ.shape, scale=np.abs(power * impedance))
    clutter = max_IQ * 10**(clutterdB / 20)
    ampl_clutter = max_IQ * 10**((amplCullerdB + clutterdB) / 20)
    return IQ + ndimage.gaussian_filter(clutter + noise * ampl_clutter, sigma=sigmaGauss)

# ==========================================
# 2. FONCTION DE CONVERSION M�TRES -> PIXELS
# ==========================================
def meters_to_pixels(pos_m, axis_m):
    """
    Convertit une position en m�tres vers un index pixel 
    en se basant sur le vecteur d'axe (P.x ou P.z).
    """
    # On trouve l'index le plus proche dans le vecteur d'axe
    return (np.abs(axis_m - pos_m)).argmin()

# ==========================================
# 3. G�N�RATION DU DATASET
# ==========================================
def prepare_full_yolo_dataset():
    for path in DIRS.values(): os.makedirs(path, exist_ok=True)
    
    NoiseParam = {"power": -2, "impedance": 0.2, "sigmaGauss": 1.5, "clutterdB": -20, "amplCullerdB": 10}
    
    for i in range(1, 21):
        file_name = f"PALA_InSilicoFlow_IQ{i:03d}.mat"
        full_path = os.path.join(DATA_PATH, file_name)
        if not os.path.isfile(full_path): continue
            
        print(f"Traitement de {file_name}...")
        mat_data = scipy.io.loadmat(full_path)
        
        # --- R�CUP�RATION DES AXES (Plus robuste) ---
        try:
            # Essai 1: Structure P standard (P['x'])
            x_axis = mat_data['P'][0,0]['x'].flatten()
            z_axis = mat_data['P'][0,0]['z'].flatten()
        except (ValueError, KeyError, IndexError):
            try:
                # Essai 2: Structure avec 'grid' (P['grid']['x'])
                grid = mat_data['P'][0,0]['grid'][0,0]
                x_axis = grid['x'].flatten()
                z_axis = grid['z'].flatten()
            except:
                # Essai 3: Variables � la racine du fichier .mat
                if 'x' in mat_data and 'z' in mat_data:
                    x_axis = mat_data['x'].flatten()
                    z_axis = mat_data['z'].flatten()
                else:
                    print(f"Erreur : Impossible de trouver les axes x et z dans {file_name}")
                    print("Champs dispos :", mat_data.keys())
                    continue
        
        iq_raw_full = np.abs(mat_data["IQ"])
        listpos_raw = mat_data["ListPos"]
        num_frames = listpos_raw.shape[2]

        for frame_idx in range(0, num_frames, 10):
            # --- IMAGE ---
            iq_frame = iq_raw_full[:, :, frame_idx]
            noisy_iq = np.abs(PALA_AddNoiseInIQ(iq_frame, **NoiseParam))
            log_iq = 20 * np.log10(np.maximum(noisy_iq, 1e-8))
            img_png = cv2.normalize(log_iq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = img_png.shape
            
            base_name = f"IQ{i:03d}_frame{frame_idx:04d}"
            img_save_path = os.path.join(DIRS["images_train"], f"{base_name}.png")
            cv2.imwrite(img_save_path, img_png)
            
            # --- LABELS & DEBUG ---
            img_debug = cv2.cvtColor(img_png, cv2.COLOR_GRAY2BGR)
            label_path = os.path.join(DIRS["labels_train"], f"{base_name}.txt")
            
            with open(label_path, 'w') as f_out:
                for bull_idx in range(listpos_raw.shape[0]):
                    # Positions physiques (en m�tres)
                    x_m = listpos_raw[bull_idx, 0, frame_idx]
                    z_m = listpos_raw[bull_idx, 2, frame_idx]
                    
                    if np.isnan(x_m) or np.isnan(z_m): continue

                    # CONVERSION : M�tres -> Index Pixel
                    x_px = meters_to_pixels(x_m, x_axis)
                    z_px = meters_to_pixels(z_m, z_axis)

                    # Normalisation pour YOLO (0 � 1)
                    x_center_norm = x_px / w
                    y_center_norm = z_px / h
                    w_norm = BOX_SIZE_PIXELS / w
                    h_norm = BOX_SIZE_PIXELS / h
                    
                    # S�curit� : on ignore si c'est hors image
                    if 0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1:
                        f_out.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        
                        # Dessin Debug
                        x1, y1 = int(x_px - BOX_SIZE_PIXELS/2), int(z_px - BOX_SIZE_PIXELS/2)
                        x2, y2 = int(x_px + BOX_SIZE_PIXELS/2), int(z_px + BOX_SIZE_PIXELS/2)
                        # cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 1) # Vert pour changer !

            cv2.imwrite(os.path.join(DIRS["debug"], f"{base_name}_debug.png"), img_debug)

if __name__ == "__main__":
    prepare_full_yolo_dataset()
    print("Fini ! Les bo�tes devraient �tre centr�es maintenant.")