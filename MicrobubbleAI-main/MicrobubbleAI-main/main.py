# -*- coding: utf-8 -*-
import torch
from dataclasses import dataclass
from load_data import load_dataset
from networks.UnetPosition import UnetPosition
from networks.UnetBulle import UnetBulle
from networks.UnetMap import UnetMap
import train as tr
import torch.nn as nn
import os

# --- CONFIGURATION DES CHEMINS ---
base_save_path = "/projects/ecptr/Espace_GIT/Espace_Clement/Chef-D-oeuvre-ECPTR-/MicrobubbleAI-main/MicrobubbleAI-main/Save"

@dataclass
class Args():
    pathData: str # chemin vers les donnees
    pathSave: str # dossier de sauvegarde (sera mis a jour dynamiquement)
    device: torch.device 
    testSize: float # pourcentage test [0;1]
    batchSize: int 
    numWorkers: int 
    shuffle: bool 
    weightDecay: float
    epochs: int
    trainType: int # 0=pos, 1=nbBulles, 2=posMap, 3=localisation
    loss: torch.nn.Module 
    std: float # ecart type heatmap

# Initialisation des arguments
args = Args(pathData = "/projects/ecptr/Dataset2D/PALA_data/PALA_data_InSilicoFlow",
            pathSave = base_save_path, # Provisoire
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            testSize = 0.15,
            batchSize = 16,
            numWorkers = 1,
            shuffle = True,
            weightDecay = 0.01,
            epochs = 20,
            trainType = 3,
            loss = nn.MSELoss(),
            std = 1.2)

# --- LOGIQUE DE DOSSIERS DYNAMIQUE ---
if args.trainType == 2:
    args.pathSave = os.path.join(base_save_path, "Type2_Segmentation")
elif args.trainType == 3:
    args.pathSave = os.path.join(base_save_path, "Type3_Heatmap")
else:
    args.pathSave = os.path.join(base_save_path, f"Type_{args.trainType}")

if not os.path.exists(args.pathSave):
    os.makedirs(args.pathSave, exist_ok=True)
    print(f"Dossier de travail cree : {args.pathSave}")

# Chargement du dataset
train_loader, test_loader, origin, data_size, max_bulles = load_dataset(args)

# --- EXECUTION DE L'ENTRAINEMENT ---

if args.trainType == 0:
    model = UnetPosition(max_bulles)
    model = model.to(args.device)
    tr.train_position_model(model, args, train_loader, test_loader, origin, data_size, max_bulles)

elif args.trainType == 1:
    model = UnetBulle()
    model = model.to(args.device)
    tr.train_bulle_model(model, args, train_loader, test_loader, max_bulles)

elif args.trainType == 2:
    model = UnetMap()
    model = model.to(args.device)
    # Sauvegarde dans /Save/Type2_Segmentation
    tr.train_position_map_model(model, args, train_loader, test_loader)

else: # Mode Heatmap (Type 3)
    # On va chercher le GUIDE dans le dossier du Type 2
    path_du_guide = os.path.join(base_save_path, "Type2_Segmentation")
    
    filename_map = None
    if os.path.exists(path_du_guide):
        files = [f for f in os.listdir(path_du_guide) if f.startswith('epoch_') and f.endswith('.pt')]
        if files:
            # On recupere la derniere epoque disponible du Type 2
            epochs_indices = [int(f.split('_')[1].split('.')[0]) for f in files]
            latest_epoch = max(epochs_indices)
            filename_map = os.path.join(path_du_guide, f"epoch_{latest_epoch}.pt")
    
    if filename_map and os.path.exists(filename_map):
        print(f"Chargement du GUIDE (Type 2) depuis : {filename_map}")
        checkpoint_map = torch.load(filename_map, map_location=args.device)
        model_map = UnetMap()
        model_map.load_state_dict(checkpoint_map['model_state_dict'])
    else:
        print("ATTENTION : Aucun guide (Type 2) trouve. Entrainement a partir d'un modele vierge.")
        model_map = UnetMap()
        
    model_map = model_map.to(args.device)
    
    # On cree le modele Type 3 (Heatmap)
    model = UnetMap()
    model = model.to(args.device)
    # Sauvegarde dans /Save/Type3_Heatmap (n'ecrasera pas le Type 2)
    tr.train_heatmap_model(model, model_map, args, train_loader, test_loader, origin, data_size)