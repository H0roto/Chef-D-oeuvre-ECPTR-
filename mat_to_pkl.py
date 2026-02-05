import scipy.io
import pickle
import os
from pathlib import Path

# Pour les fichiers MATLAB v7.3 (HDF5)
try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_mat_file(filepath: str):
    """
    Charge un fichier .mat (supporte v7.3 et versions antérieures)
    """
    try:
        # Essayer d'abord avec scipy (versions < 7.3)
        return scipy.io.loadmat(filepath)
    except NotImplementedError:
        # Fichier v7.3, utiliser mat73 ou h5py
        if HAS_MAT73:
            return mat73.loadmat(filepath)
        elif HAS_H5PY:
            # Lecture manuelle avec h5py
            data = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][()]
            return data
        else:
            raise ImportError("Installez mat73 ou h5py pour lire les fichiers MATLAB v7.3: pip install mat73")


def mat_to_pickle(input_folder: str, output_folder: str = None):
    """
    Convertit tous les fichiers .mat d'un dossier en fichiers .pkl
    
    Args:
        input_folder: Chemin du dossier contenant les fichiers .mat
        output_folder: Chemin du dossier de sortie (optionnel, par défaut = input_folder)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder) if output_folder else input_path
    
    # Créer le dossier de sortie s'il n'existe pas
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trouver tous les fichiers .mat
    mat_files = list(input_path.glob("*.mat"))
    
    if not mat_files:
        print(f"Aucun fichier .mat trouvé dans {input_folder}")
        return
    
    print(f"Conversion de {len(mat_files)} fichier(s) .mat...")
    
    for mat_file in mat_files:
        try:
            # Charger le fichier .mat (supporte v7.3)
            mat_data = load_mat_file(str(mat_file))
            
            # Créer le chemin du fichier .pkl
            pkl_file = output_path / (mat_file.stem + ".pkl")
            
            # Sauvegarder en pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump(mat_data, f)
            
            print(f"✓ {mat_file.name} -> {pkl_file.name}")
            
        except Exception as e:
            print(f"✗ Erreur avec {mat_file.name}: {e}")
    
    print("Conversion terminée!")


if __name__ == "__main__":
    # Exemple d'utilisation:
    # Convertir les fichiers .mat du dossier ULM3D_data
    input_folder = r"ULM3D_data"
    output_folder = r"ULM3D_data_pkl"  # Optionnel: dossier de sortie différent
    
    mat_to_pickle(input_folder, output_folder)
