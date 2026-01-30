# -*- coding: utf-8 -*-
from networks.UnetBulle import UnetBulle
from networks.UnetPosition import UnetPosition
from networks.UnetMap import UnetMap
import torch
from torchvision import transforms
import scipy.io
import os
from os.path import join
import numpy as np
from scipy import ndimage
from torch.nn.functional import normalize
import json
import matplotlib.pyplot as plt
import util as ut
import cv2
from scipy import signal
import torch.nn.functional as F

#TO ADAPT
nbSamples = 10

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
        max_IQ = np.max(IQ)
        return IQ + ndimage.gaussian_filter(max_IQ * 10**(clutterdB / 20) + np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape) * max_IQ * 10**((amplCullerdB + clutterdB) / 20), sigma=sigmaGauss)

def read_flow_data(pathData):
    transform = transforms.ToTensor()
    sequence = scipy.io.loadmat(join(pathData,"PALA_InSilicoFlow_sequence.mat"))
    Origin = sequence["PData"]["Origin"].flatten()[0][0]
    data_size = sequence["PData"]["Size"].flatten()[0][0]
    NoiseParam = {}
    NoiseParam["power"]        = -2;   # [dBW]
    NoiseParam["impedance"]    = .2;   # [ohms]
    NoiseParam["sigmaGauss"]   = 1.5;  # Gaussian filtering
    NoiseParam["clutterdB"]    = -20;  # Clutter level in dB (will be changed later)
    NoiseParam["amplCullerdB"] = 10;   # dB amplitude of clutter
    IQs, xy_pos, max_bulles = None, None, None
    for file in os.listdir(join(pathData,"IQ")):
        temp = scipy.io.loadmat(join(pathData,"IQ",file))
        if max_bulles is None:
            max_bulles = temp["ListPos"].shape[0]
        xy_pos = torch.cat((xy_pos, transform(temp["ListPos"][:,[0, 2],:])), dim=0) if xy_pos is not None else transform(temp["ListPos"][:,[0, 2],:])
        IQs = torch.cat((IQs, transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))), dim=0) if IQs is not None else transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))
    return normalize(IQs), xy_pos, Origin, data_size, max_bulles

def SVDfilter(IQ,cutoff):
	initsize = IQ.shape
	initsize_x = initsize[0]
	initsize_y = initsize[1]
	initsize_z = initsize[2]
	if cutoff[-1] > initsize[-1]:
		cutoff = [cutoff[0],initsize[-1]]
	if len(cutoff) == 1:
		cutoff = [cutoff[0],initsize[-1]]
	elif len(cutoff) == 2:
		cutoff = [cutoff[0],cutoff[1]] 
	if cutoff == [1,IQ.shape[2]] or cutoff[0] < 2:
		IQf = IQ
		return IQf
	cutoff[0] = cutoff[0] - 1
	cutoff[1] = cutoff[1] - 1 # in MatLab, array[200] give you access to the 200th element, unlike Python
	X = np.reshape(IQ,(initsize_x*initsize_y,initsize_z))# % Reshape into Casorati matric
	U,S,Vh = scipy.linalg.svd(np.dot(X.T,X)) #calculate svd of the autocorrelated Matrix
	V = np.dot(X,U) # Calculate the singular vectors.
	Reconst = np.dot(V[:,cutoff],U[:,cutoff].T) # Singular value decomposition
	IQf = np.reshape(Reconst,(initsize_x,initsize_y,initsize_z)) #Reconstruction of the final filtered matrix
	return np.absolute(IQf)

def read_vivo_data(pathData):
    transform = transforms.ToTensor()
    IQs = None
    for file in os.listdir(join(pathData,"IQ")):
        temp = scipy.io.loadmat(join(pathData,"IQ",file))
        IQ = temp["IQ"]
        framerate = temp["UF"]["FrameRateUF"][0][0][0][0]
        cutoff = [50,framerate]
        bulles = SVDfilter(IQ,cutoff)
        but_b,but_a = scipy.signal.butter(2,[50/(framerate*0.5),249/(framerate*0.5)],btype='bandpass')
        bulles = signal.lfilter(but_b,but_a,bulles,axis=2)
        IQs = torch.cat((IQs, transform(bulles)), dim=0) if IQs is not None else transform(bulles)
    return normalize(IQs)

def temp_read_vivo_data(pathData):
    transform = transforms.ToTensor()
    IQs = None
    for file in os.listdir(join(pathData,"IQ")):
        temp = scipy.io.loadmat(join(pathData,"IQ",file))
        IQ = temp["IQ_filt"]
        framerate = 1000
        cutoff = [50,framerate]
        bulles = SVDfilter(IQ,cutoff)
        but_b,but_a = scipy.signal.butter(2,[50/(framerate*0.5),249/(framerate*0.5)],btype='bandpass')
        bulles = signal.lfilter(but_b,but_a,bulles,axis=2)
        
        bulles = np.nan_to_num(bulles) 
        
        IQs = torch.cat((IQs, transform(bulles)), dim=0) if IQs is not None else transform(bulles)
    return normalize(IQs)

def temp_in_vivo_inference():
    # 1. Définition manuelle des chemins des modeles 
    filename_heatmap = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/save/epoch_heatmap.pt"
    filename_map = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/save/epoch_map.pt"
    
    # 2. Définition manuelle des dossiers de données et de sauvegarde
    pathData = "/projects/ecptr/Dataset2D/PALA_data/PALA_data_InSilicoFlow"
    pathSave = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/results/"

    # --- CHARGEMENT DU MODELE HEATMAP ---
    print(f"Chargement du modele Heatmap : {filename_heatmap}")
    checkpoint_heatmap = torch.load(filename_heatmap, map_location=device)
    if checkpoint_heatmap['model_name'] == 'UnetHeatmap' or checkpoint_heatmap['model_name'] == 'UnetMap':
        model_heatmap = UnetMap() # Ou UnetHeatmap selon ta classe
        model_heatmap.load_state_dict(checkpoint_heatmap['model_state_dict'])
    else:
        raise Exception(f"Le modele {filename_heatmap} n'est pas un UnetHeatmap")

    # --- CHARGEMENT DU MODELE MAP ---
    print(f"Chargement du modele Map : {filename_map}")
    checkpoint_map = torch.load(filename_map, map_location=device)
    if checkpoint_map['model_name'] == 'UnetMap':
        model_map = UnetMap()
        model_map.load_state_dict(checkpoint_map['model_state_dict'])
    else:
        raise Exception(f"Le modele {filename_map} n'est pas un UnetMap")

    # Préparation des modeles
    model_heatmap.eval()
    model_heatmap.to(device)
    model_map.eval()
    model_map.to(device)

    # --- TRAITEMENT ---
    print("Lecture des donnees In Vivo...")
    IQs = temp_read_vivo_data(pathData)
    
    padding = (0, 25, 0, 6)
    result_dict = {}
    
    print(f"Debut de l inference sur {len(IQs)} images...")
    for i in range(min(len(IQs), nbSamples)):
        img_tensor = IQs[i].to(device=device, dtype=torch.float)
        
        # Ajout des dimensions batch et channel [1, 1, H, W]
        img_input = img_tensor.unsqueeze(0).unsqueeze(0)
        
        # Application du padding
        img_input = F.pad(img_input, padding, "constant", 0)
        
        with torch.no_grad(): 
            pos_prediction_raw = model_heatmap(img_input)
            out_probability_raw = model_map(img_input)
        
        # Suppression du padding sur la sortie pour retrouver la taille originale
        # On retire 25 en bas (H) et 6 à droite (W)
        pos_prediction = torch.squeeze(pos_prediction_raw)[:, :img_tensor.shape[0], :img_tensor.shape[1]].cpu().numpy()
        out_probability_img = torch.squeeze(out_probability_raw)[:, :img_tensor.shape[0], :img_tensor.shape[1]].cpu().numpy()
        
        pos_prediction = ut.heatmap_to_coordinates(pos_prediction, out_probability_img)
        result_dict[f'pred_position_img{i}'] = pos_prediction.tolist()

    # Sauvegarde du JSON
    os.makedirs(pathSave, exist_ok=True) # Crée le dossier s'il n'existe pas
    output_path = os.path.join(pathSave, 'result.json')
    with open(output_path, 'w') as json_file:
        json.dump(result_dict, json_file)
    
    print(f"Inference terminee. Resultats sauvegardes dans : {output_path}")


def ask_data_and_save_path():
    """ 
    Remplacement de filedialog par des chemins fixes pour Slurm car ca crashe sinon
    """
    pathData = "/projects/ecptr/Dataset2D/PALA_data/PALA_data_InSilicoFlow"
    pathSave = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/results/"
    os.makedirs(pathSave, exist_ok=True)
    return pathData, pathSave

def get_latest_checkpoint(folder_path):
    files = [f for f in os.listdir(folder_path) if f.startswith('epoch_') and f.endswith('.pt')]
    if not files:
        raise Exception(f"Aucun checkpoint trouve dans {folder_path}")
    # On extrait le nombre, on trie et on prend le max
    latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in files])
    return os.path.join(folder_path, f"epoch_{latest_epoch}.pt")

def unet_inference():
    base_models_path = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/MicrobubbleAI-main/MicrobubbleAI-main/Save"
    filename_pos = get_latest_checkpoint(os.path.join(base_models_path, "Type0_Position"))
    filename_nbbulles = get_latest_checkpoint(os.path.join(base_models_path, "Type1_Bulle"))    
    pathData, pathSave = ask_data_and_save_path()

    checkpoint_pos = torch.load(filename_pos, map_location=device)
    model_pos = None
    if checkpoint_pos['model_name'] == 'UnetPosition':
        model_pos = UnetPosition(checkpoint_pos['max_bulles'])
        model_pos.load_state_dict(checkpoint_pos['model_state_dict'])
    else:
        raise Exception("Vous n'avez pas choisi un modÃ¨le correct pour effectuer la prÃ©diction des coordonÃ©es") 

    checkpoint_nbbulles = torch.load(filename_nbbulles, map_location=device)
    model_nbbulles, nbMaxBulles = None, None
    if checkpoint_nbbulles['model_name'] == 'UnetBulle':
        nbMaxBulles = checkpoint_nbbulles['max_bulles']
        model_nbbulles = UnetBulle()
        model_nbbulles.load_state_dict(checkpoint_nbbulles['model_state_dict'])
    else:
        raise Exception("Vous n'avez pas choisi un modÃ¨le correct pour effectuer la dÃ©tection du nombre de bulles") 

    model_pos.eval()
    model_pos.to(device)
    model_nbbulles.eval()
    model_nbbulles.to(device)
    IQs, xy_pos, origin, data_size, max_bulles = read_flow_data(pathData)
    random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
    result_dict = {}
    for i, num in enumerate(random_samples):
        img_tensor = IQs[num].to(device=device, dtype=torch.float)
        pos_prediction = model_pos(torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0))
        pos_prediction = torch.squeeze(pos_prediction).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(pos_prediction).detach().numpy()
        nbbulles_prediction = model_nbbulles(torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0)) * nbMaxBulles
        nbbulles_prediction = np.round(nbbulles_prediction.cpu().detach().numpy()) if device==torch.device("cuda") else np.round(nbbulles_prediction.detach().numpy())
        img_numpy = img_tensor.cpu().detach().numpy() if device==torch.device("cuda") else img_tensor.detach().numpy()
        pos_prediction[:, 0] *= data_size[1]
        pos_prediction[:, 1] *= data_size[0]
        #ground_truth = xy_pos[num].clone()
        #ground_truth = ground_truth[torch.isfinite(ground_truth)]
        #ground_truth = torch.reshape(ground_truth, (-1, 2))
        #ground_truth[:, 0] = ground_truth[:, 0] - origin[0]
        #ground_truth[:, 1] = ground_truth[:, 1] - origin[2]
        #ground_truth = ground_truth[~torch.any(ground_truth<0, axis=1)] #enlÃ¨ve les valeurs infÃ©rieures Ã  0
        #ground_truth = ground_truth[torch.logical_and(ground_truth[:, 0] <= data_size[1], ground_truth[:, 1] <= data_size[0])] #enlÃ¨ve les valeurs supÃ©rieures aux bordures de l'image
        #print(f"img nÂ°{i}: {ground_truth}")
        coordonnees = pos_prediction[:int(nbbulles_prediction),:].tolist()
        plt.imshow(img_numpy, cmap='gray')
        plt.scatter(*zip(*coordonnees), color='red', marker='x', label='Predictions')  # Afficher les coordonnÃ©es avec des croix rouges
        plt.legend()
        plt.savefig(pathSave + f"img_{i}.png")
        plt.clf()
        result_dict[f'pred_nbBulles_img{i}'] = int(nbbulles_prediction)
        result_dict[f'pred_position_img{i}'] = coordonnees
    with open(pathSave + 'result.json', 'w') as json_file:
        json.dump(result_dict, json_file)

def map_inference():
    base_models_path = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/MicrobubbleAI-main/MicrobubbleAI-main/Save"
    filename_map = get_latest_checkpoint(os.path.join(base_models_path, "Type2_Segmentation"))    
    print("Modele choisi: ", filename_map)
    pathData, pathSave = ask_data_and_save_path()
    checkpoint_map = torch.load(filename_map, map_location=device)
    if checkpoint_map['model_name'] == 'UnetMap':
        model_map = UnetMap()
        model_map.load_state_dict(checkpoint_map['model_state_dict'])
    else:
        raise Exception("Veuillez choisir un modÃ¨le Map pour effectuer la prÃ©diction des coordonnÃ©es")
    model_map.eval()
    model_map.to(device)
    IQs, xy_pos, origin, data_size, _ = read_flow_data(pathData)
    random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
    for i, num in enumerate(random_samples):
        img_tensor = IQs[num].to(device=device, dtype=torch.float)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        ground_truth = xy_pos[num].clone()
        pos_prediction = model_map(torch.unsqueeze(img_tensor, 0))
        pos_prediction = torch.squeeze(pos_prediction).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(pos_prediction).detach().numpy()
        ground_truth = ut.coordinates_to_mask(torch.unsqueeze(ground_truth, 0), img_tensor.shape, origin, data_size)
        ground_truth = torch.squeeze(ground_truth).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(ground_truth).detach().numpy()
        img_prediction = cv2.normalize(pos_prediction, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        img_ground_truth = cv2.normalize(ground_truth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(pathSave + f"predicted_img{i}.png", img_prediction)
        cv2.imwrite(pathSave + f"ground_truth_img{i}.png", img_ground_truth)

def heatmap_inference():
    # --- CONFIGURATION DES CHEMINS ---
    base_save_path = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/MicrobubbleAI-main/MicrobubbleAI-main/Save"
    path_type2 = os.path.join(base_save_path, "Type2_Segmentation")
    path_type3 = os.path.join(base_save_path, "Type3_Heatmap")
    
    # Dossier pour les images de sortie
    path_results = os.path.join(base_save_path, "Results_Inference")
    if not os.path.exists(path_results):
        os.makedirs(path_results, exist_ok=True)

    # --- RECHERCHE AUTOMATIQUE DES MODELES ---
    try:
        # Trouver la derniere époque du Type 2 (le Guide/Map)
        files_2 = [f for f in os.listdir(path_type2) if f.startswith('epoch_') and f.endswith('.pt')]
        latest_2 = max([int(f.split('_')[1].split('.')[0]) for f in files_2])
        filename_map = os.path.join(path_type2, f"epoch_{latest_2}.pt")

        # Trouver la derniere époque du Type 3 (la Heatmap)
        files_3 = [f for f in os.listdir(path_type3) if f.startswith('epoch_') and f.endswith('.pt')]
        latest_3 = max([int(f.split('_')[1].split('.')[0]) for f in files_3])
        filename_heatmap = os.path.join(path_type3, f"epoch_{latest_3}.pt")
    except Exception as e:
        print(f"Erreur lors de la recherche des modeles : {e}")
        print("Verifie que les dossiers Type2_Segmentation et Type3_Heatmap contiennent bien des fichiers .pt")
        return

    print(f"Prediction Heatmap avec :\n > Map: {filename_map}\n > Heatmap: {filename_heatmap}")

    # --- CHARGEMENT DES MODELES ---
    checkpoint_heatmap = torch.load(filename_heatmap, map_location=device)
    model_heatmap = UnetMap().to(device)
    model_heatmap.load_state_dict(checkpoint_heatmap['model_state_dict'])
    model_heatmap.eval()

    checkpoint_map = torch.load(filename_map, map_location=device)
    model_map = UnetMap().to(device)
    model_map.load_state_dict(checkpoint_map['model_state_dict'])
    model_map.eval()

    # --- CHARGEMENT DES DONNÉES ---
    pathData = "/projects/ecptr/Dataset2D/PALA_data/PALA_data_InSilicoFlow"
    IQs, xy_pos, origin, data_size, _ = read_flow_data(pathData)
    
    random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
    result_dict = {}

    for i, num in enumerate(random_samples):
        img_tensor = IQs[num].to(device=device, dtype=torch.float)
        img_input = torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0) # Format [1, 1, H, W]
        
        # 1. Prediction de la heatmap (Type 3)
        pos_prediction_heatmap = model_heatmap(img_input)
        pos_prediction_heatmap = torch.squeeze(pos_prediction_heatmap).cpu().detach().numpy()
        
        # 2. Prediction de la map de probabilité (Type 2)
        out_probability_img = model_map(img_input)
        out_probability_img = torch.squeeze(out_probability_img).cpu().detach().numpy()
        
        # 3. Conversion heatmap -> coordonnées (x,y) en utilisant le guide
        pos_prediction = ut.heatmap_to_coordinates(pos_prediction_heatmap, out_probability_img)
        
        # 4. Preparation de la Vérité Terrain (Ground Truth)
        gt_save = xy_pos[num].clone()
        gt_numpy = ut.process_data(gt_save, origin, data_size).cpu().detach().numpy()

        # --- PLOT ET SAUVEGARDE ---
        plt.figure()
        plt.imshow(img_tensor.cpu().detach().numpy(), cmap='gray')
        
        if len(pos_prediction) > 0:
            plt.scatter(*zip(*pos_prediction), color='red', marker='+', label='Predictions')
        
        plt.scatter(gt_numpy[:,0], gt_numpy[:,1], color='green', marker='x', label='Verite terrain')
        
        plt.title(f"Image {num} - Inference Heatmap")
        plt.legend(loc='best')
        plt.savefig(os.path.join(path_results, f"img_heatmap_{i}.png"))
        plt.close() # Important pour ne pas saturer la mémoire

        result_dict[f'ground_truth_img{i}'] = gt_numpy.tolist()
        result_dict[f'pred_position_img{i}'] = pos_prediction.tolist()

    # Sauvegarde des résultats numériques
    with open(os.path.join(path_results, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False)
    
    print(f"Inference terminee. Resultats dans : {path_results}")
 
def in_vivo_inference():
    base_models_path = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/MicrobubbleAI-main/MicrobubbleAI-main/Save"
    
    filename_heatmap = get_latest_checkpoint(os.path.join(base_models_path, "Type3_Heatmap"))
    filename_map = get_latest_checkpoint(os.path.join(base_models_path, "Type2_Segmentation"))

    # 1. Charger Heatmap
    checkpoint_heatmap = torch.load(filename_heatmap, map_location=device)
    model_heatmap = UnetMap().to(device)
    model_heatmap.load_state_dict(checkpoint_heatmap['model_state_dict'])

    # 2. Charger Map
    checkpoint_map = torch.load(filename_map, map_location=device)
    model_map = UnetMap().to(device)
    model_map.load_state_dict(checkpoint_map['model_state_dict'])

    pathData, pathSave = ask_data_and_save_path()
    
    model_heatmap.eval()
    model_heatmap.to(device)
    model_map.eval()
    model_map.to(device)
    IQs = read_vivo_data(pathData)
    random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
    transform = transforms.Resize((84,143))
    for i, num in enumerate(random_samples):
        img_tensor = IQs[num].to(device=device, dtype=torch.float)
        img_tensor = transform(torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0))
        pos_prediction = model_heatmap(img_tensor)
        pos_prediction = torch.squeeze(pos_prediction).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(pos_prediction).detach().numpy()
        out_probability_img = model_map(img_tensor)
        out_probability_img = torch.squeeze(out_probability_img).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(out_probability_img).detach().numpy()
        pos_prediction = ut.heatmap_to_coordinates(pos_prediction, out_probability_img)
        plt.imshow(torch.squeeze(img_tensor).cpu().detach().numpy(), cmap='gray')
        plt.scatter(*zip(*pos_prediction), color='red', marker='+', label='Predictions')
        plt.legend(loc='best')
        plt.savefig(pathSave + f"img_{i}.png")
        plt.clf()
        plt.imshow(torch.squeeze(img_tensor).cpu().detach().numpy(), cmap='gray')
        plt.savefig(pathSave + f"orig_img{i}.png")
        plt.clf()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Tk().withdraw()

def silico_flow_data():
    while True:
        print("Choisissez le type d'inference:")
        print('1. PrÃ©diction du nombre de bulles + coordonnees en parallele')
        print('2. (position map) Localisation par extraction d\'information sur les images')
        print('3. Localisation par heatmap')
        choice_train_model = input('Entrez votre choix: ')
        print('')
        if choice_train_model == '1':
            unet_inference()
            break
        elif choice_train_model == '2':
            map_inference()
            break
        elif choice_train_model == '3':
            heatmap_inference()
            break
        print('Ce choix est invalide, veuillez choisir un nombre entre 1-2.')

while True:
    print("Choisissez le type de donnÃ©es:")
    print('1. In Silico Flow')
    print('2. In Vivo')
    choice_train_model = input('Entrez votre choix: ')
    print('')
    if choice_train_model == '1':
        silico_flow_data()
        break
    elif choice_train_model == '2':
        temp_in_vivo_inference()
        break
    print('Ce choix est invalide, veuillez choisir un nombre entre 1-2.')
