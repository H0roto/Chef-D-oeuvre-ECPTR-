import scipy.io
import torch
import numpy as np
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


def load_mat_file(filepath: str) -> dict:
    """
    Charge un fichier .mat (supporte v7.3 et versions antérieures)
    """
    try:
        # Essayer d'abord avec scipy (versions < 7.3)
        return scipy.io.loadmat(filepath, squeeze_me=True)
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


def convert_to_tensor(data):
    """
    Convertit récursivement les arrays numpy en tensors PyTorch.
    Gère les dictionnaires, listes et arrays numpy.
    """
    if isinstance(data, np.ndarray):
        # Ignorer les arrays de type object ou string
        if data.dtype.kind in ['O', 'U', 'S']:
            return data
        # Convertir en float32 pour optimisation GPU
        if np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data.astype(np.float32))
        elif np.issubdtype(data.dtype, np.complexfloating):
            # PyTorch supporte les complexes
            return torch.from_numpy(data.astype(np.complex64))
        elif np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data.astype(np.int64))
        else:
            return torch.from_numpy(data)
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_tensor(item) for item in data]
    else:
        return data


def mat_to_pt(input_folder: str, output_folder: str = None, var_name: str = None):
    """
    Convertit tous les fichiers .mat d'un dossier en fichiers .pt (PyTorch)
    
    Args:
        input_folder: Chemin du dossier contenant les fichiers .mat
        output_folder: Chemin du dossier de sortie (optionnel, par défaut = input_folder)
        var_name: Nom de la variable à extraire (optionnel, si None garde tout le dict)
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
    
    print(f"Conversion de {len(mat_files)} fichier(s) .mat vers .pt...")
    
    for mat_file in mat_files:
        try:
            # Charger le fichier .mat
            mat_data = load_mat_file(str(mat_file))
            
            # Filtrer les métadonnées MATLAB (commençant par __)
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            
            # Extraire une variable spécifique si demandé
            if var_name and var_name in mat_data:
                data_to_save = {var_name: convert_to_tensor(mat_data[var_name])}
            else:
                data_to_save = convert_to_tensor(mat_data)
            
            # Créer le chemin du fichier .pt
            pt_file = output_path / (mat_file.stem + ".pt")
            
            # Sauvegarder en format PyTorch
            torch.save(data_to_save, pt_file)
            
            # Afficher les infos
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            print(f"✓ {mat_file.name} -> {pt_file.name} ({size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"✗ Erreur avec {mat_file.name}: {e}")
    
    print("Conversion terminée!")


def load_pt(pt_file: str, var_name: str = None, device: str = "cpu") -> torch.Tensor:
    """
    Charge un fichier .pt et retourne le tensor (optionnel: directement sur GPU)
    
    Args:
        pt_file: Chemin du fichier .pt
        var_name: Nom de la variable à extraire
        device: 'cpu' ou 'cuda' pour charger directement sur GPU
    
    Returns:
        Tensor PyTorch
    """
    data = torch.load(pt_file, map_location=device, weights_only=False)
    
    if var_name and isinstance(data, dict):
        return data[var_name]
    return data


def verify_conversion(mat_folder: str, pt_folder: str, var_name: str = None, rtol: float = 1e-5, atol: float = 1e-5):
    """
    Vérifie que les fichiers .mat et .pt ont le même contenu.
    
    Args:
        mat_folder: Chemin du dossier contenant les fichiers .mat originaux
        pt_folder: Chemin du dossier contenant les fichiers .pt convertis
        var_name: Nom de la variable à comparer (optionnel)
        rtol: Tolérance relative pour la comparaison
        atol: Tolérance absolue pour la comparaison
    
    Returns:
        bool: True si tous les fichiers sont identiques, False sinon
    """
    mat_path = Path(mat_folder)
    pt_path = Path(pt_folder)
    
    mat_files = list(mat_path.glob("*.mat"))
    
    if not mat_files:
        print(f"Aucun fichier .mat trouvé dans {mat_folder}")
        return False
    
    print(f"Vérification de {len(mat_files)} fichier(s)...")
    all_ok = True
    
    for mat_file in mat_files:
        pt_file = pt_path / (mat_file.stem + ".pt")
        
        if not pt_file.exists():
            print(f"✗ {mat_file.name}: Fichier .pt correspondant non trouvé")
            all_ok = False
            continue
        
        try:
            # Charger les deux fichiers
            mat_data = load_mat_file(str(mat_file))
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            pt_data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            # Si var_name spécifié, comparer uniquement cette variable
            keys_to_check = [var_name] if var_name else list(mat_data.keys())
            
            file_ok = True
            for key in keys_to_check:
                if key not in mat_data:
                    continue
                if key not in pt_data:
                    print(f"  ✗ {mat_file.name}: Clé '{key}' absente du fichier .pt")
                    file_ok = False
                    continue
                
                mat_val = mat_data[key]
                pt_val = pt_data[key]
                
                # Convertir le tensor en numpy pour comparaison
                if torch.is_tensor(pt_val):
                    pt_val = pt_val.numpy()
                
                if isinstance(mat_val, np.ndarray) and isinstance(pt_val, np.ndarray):
                    # Vérifier les shapes
                    if mat_val.shape != pt_val.shape:
                        print(f"  ✗ {mat_file.name} [{key}]: Shapes différentes - mat:{mat_val.shape} vs pt:{pt_val.shape}")
                        file_ok = False
                        continue
                    
                    # Comparer les valeurs (avec tolérance pour float32)
                    if np.allclose(mat_val, pt_val, rtol=rtol, atol=atol):
                        pass  # OK
                    else:
                        diff = np.abs(mat_val - pt_val).max()
                        print(f"  ✗ {mat_file.name} [{key}]: Différence max = {diff}")
                        file_ok = False
            
            if file_ok:
                print(f"✓ {mat_file.name}: Identique")
            else:
                all_ok = False
                
        except Exception as e:
            print(f"✗ Erreur avec {mat_file.name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ Tous les fichiers sont identiques!")
    else:
        print("\n✗ Des différences ont été détectées.")
    
    return all_ok


if __name__ == "__main__":
    # Exemple d'utilisation:
    # Convertir les fichiers .mat du dossier ULM3D_data
    input_folder = r"Open3DULM/ULM3D_data"
    output_folder = r"Open3DULM/ULM3D_data_pt"
    
    # Optionnel: spécifier le nom de la variable IQ à extraire
    var_name = "IQ"  # Mettre None pour garder toutes les variables
    
    mat_to_pt(input_folder, output_folder, var_name=var_name)
    
    # Vérifier que la conversion est correcte
    #print("\n--- Vérification de la conversion ---")
    #verify_conversion(input_folder, output_folder, var_name=var_name)
