#!/bin/bash
# Script d'installation et d'exécution de MVSEP-MDX23-music-separation-model
# Ce script installe les dépendances, télécharge les modèles nécessaires et exécute l'inférence

# Récupération du nom d'utilisateur actuel
CURRENT_USER=$(whoami)
echo "Exécution du script en tant que: $CURRENT_USER"

# Mise à jour des dépôts et installation des paquets de base
echo "Installation des dépendances système..."
sudo apt update
sudo apt install git -y
sudo apt install git-lfs -y
git lfs install

# Clonage du dépôt
echo "Clonage du dépôt depuis HuggingFace..."
git clone https://huggingface.co/spaces/Yeluo0204/MVSEP-MDX23-music-separation-model
cd MVSEP-MDX23-music-separation-model/

# Installation des dépendances Python
echo "Installation des dépendances Python..."
sudo apt update
sudo apt install python3-pip -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --force-reinstall
python3 -m pip install html5lib
python3 -m pip install -r requirements.txt
python3 -m pip install boto3
python3 -m pip install pynvml
sudo apt install -y libsndfile1

# Création du répertoire des modèles
echo "Création du répertoire des modèles..."
mkdir -p ./models

# Téléchargement des modèles
echo "Téléchargement des modèles..."
wget -O ./models/04573f0d-f3cf25b2.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th"
wget -O ./models/Kim_Vocal_2.onnx "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx"
wget -O ./models/Kim_Inst.onnx "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Inst.onnx"

# Création des répertoires de cache et téléchargement des checkpoints
echo "Création des répertoires de cache..."
sudo mkdir -p /root/.cache/torch/hub/checkpoints
sudo mkdir -p /home/$CURRENT_USER/.cache/torch/hub/checkpoints

echo "Téléchargement des checkpoints..."
sudo wget -O /root/.cache/torch/hub/checkpoints/f7e0c4bc-ba3fe64a.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/d12395a8-e57c48e6.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/92cfc3b6-ef3bcb9c.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/04573f0d-f3cf25b2.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/955717e8-8726e21a.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/5c90dfd2-34c22ccb.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"
sudo wget -O /root/.cache/torch/hub/checkpoints/75fc33f5-1941ce65.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th"
sudo wget -O /home/$CURRENT_USER/.cache/torch/hub/checkpoints/5c90dfd2-34c22ccb.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"
sudo wget -O /home/$CURRENT_USER/.cache/torch/hub/checkpoints/75fc33f5-1941ce65.th "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th"

# Création du script patch Python
echo "Création du script patch Python..."
curl -k https://raw.githubusercontent.com/cyrille800/Genlab/main/cr.py -o cr.py
curl -k https://raw.githubusercontent.com/cyrille800/Genlab/main/inference_demucs.py -o inference_demucs.py
rm inference.py

# Création du dossier de résultats
echo "Création du répertoire des résultats..."
mkdir -p ./results

echo "Installation terminée avec succès!"
