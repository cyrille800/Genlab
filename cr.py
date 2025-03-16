#!/usr/bin/env python3
import torch
from demucs.htdemucs import HTDemucs
import torch.serialization
import sys 
import runpy
import glob
import subprocess
import math
import boto3
import threading
import time
import requests
import traceback
import pynvml
import datetime
import signal
import requests
import json
from pydub import AudioSegment
import functools  
import multiprocessing
import tempfile
import os

# Constante pour la durée maximale d'exécution (en secondes)
MAX_EXECUTION_TIME = 3000  # 1 heure par défaut, ajustez selon vos besoins

PROVIDER_POD = ""
if os.environ.get("RUNPOD_SECRET_CLOUDFARE_R2_ACCESS_KEY_ID"):
    PROVIDER_POD = "RUNPOD_SECRET_"
if os.environ.get("VASTAI_SECRET_CLOUDFARE_R2_ACCESS_KEY_ID"):
    PROVIDER_POD = "VASTAI_SECRET_"

genlab_customer_id=""
def push_kv_runpod(data):
    try:

        account_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_ACCOUNT_ID")
        kv_namespace_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_KV_NAMESPACE_ID")
        api_token = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_KV_API_TOKEN")
        
        # Headers pour l'authentification
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
        url_kv = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{kv_namespace_id}/values/{genlab_customer_id}"
    
        response = requests.put(url_kv, headers=headers, data=json.dumps({
            "step": "(1/5) extract instruments",
            "value": data
        }))

        if response.status_code!=200:
            send_discord_error("Erreur de fichier d'état", f"Erreur de mise a jour du kv dans cloudfare pour le client numero: {genlab_customer_id}")                                                      
    except Exception as e:
        print(f"Erreur de mise a jour du kv dans cloudfare: {e}")
        send_discord_error("Erreur de fichier d'état", f"Erreur de mise a jour du kv dans cloudfare: {e}", traceback.format_exc())
    
def send_discord_error(error_title, error_details, error_traceback=None):
    """
    Envoie un message d'erreur à un webhook Discord.
    
    Args:
        error_title: Titre de l'erreur
        error_details: Détails de l'erreur
        error_traceback: Traceback de l'erreur (optionnel)
    """
    try:

        webhook_url_param1 = os.environ.get(f"{PROVIDER_POD}DISCORD_WEBHOOK_URL_SALON_ERROR_PART1")
        webhook_url_param2 = os.environ.get(f"{PROVIDER_POD}DISCORD_WEBHOOK_URL_SALON_ERROR_PART2")
        webhook_url = f"https://discord.com/api/webhooks/{webhook_url_param1}/{webhook_url_param2}"
            
        # Construction du message
        message = {
            "embeds": [
                {
                    "title": f"⚠️ {error_title}",
                    "color": 16711680,  # Rouge
                    "description": error_details,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            ]
        }
        
        # Ajouter le traceback s'il est disponible
        if error_traceback:
            # Diviser le traceback en morceaux de 1000 caractères max (limite Discord)
            tb_chunks = [error_traceback[i:i+1000] for i in range(0, len(error_traceback), 1000)]
            
            message["embeds"][0]["fields"] = []
            for i, chunk in enumerate(tb_chunks):
                message["embeds"][0]["fields"].append({
                    "name": f"Traceback {i+1}/{len(tb_chunks)}" if len(tb_chunks) > 1 else "Traceback",
                    "value": f"```{chunk}```"
                })
            
        # Envoi du message au webhook
        response = requests.post(
            webhook_url,
            json=message
        )
        response.raise_for_status()
        
        print(f"Message d'erreur envoyé à Discord: {error_title}")
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'envoi du message à Discord: {e}")
        return False

# Gestion du timeout global
def timeout_handler(signum, frame):
    """Gestionnaire pour le signal de timeout"""
    error_msg = f"L'exécution du script a dépassé le délai maximum de {MAX_EXECUTION_TIME} secondes"
    print(f"Erreur: {error_msg}")
    send_discord_error("Timeout d'exécution", error_msg)
    sys.exit(1)

# Patch pour torch.load
torch.serialization.add_safe_globals([HTDemucs])
original_load = torch.load
def patched_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)
torch.load = patched_load

class ProgressPercentage:
    """Classe pour afficher la progression du téléchargement/téléversement."""
    def __init__(self, filename, is_download=False):
        self._filename = filename
        self._is_download = is_download
        if not is_download:
            # Pour le téléversement, on connaît la taille du fichier
            self._size = float(os.path.getsize(filename)) if os.path.exists(filename) else 0
        else:
            # Pour le téléchargement, on ne connaît pas encore la taille
            self._size = 0
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._last_time = time.time()
        self._last_seen = 0
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            
            # Pour le téléchargement, mettre à jour la taille totale si nécessaire
            if self._is_download and self._size == 0:
                # Essayer de mettre à jour la taille totale si le fichier existe déjà
                try:
                    self._size = float(os.path.getsize(self._filename))
                except:
                    pass
            
            percentage = (self._seen_so_far / self._size) * 100 if self._size > 0 else 0
            
            # Calcul de la vitesse
            current_time = time.time()
            elapsed = current_time - self._last_time
            if elapsed >= 1.0:  # Afficher la vitesse toutes les secondes
                speed = (self._seen_so_far - self._last_seen) / elapsed / 1024  # KB/s
                self._last_time = current_time
                self._last_seen = self._seen_so_far
                
                if self._size > 0:
                    sys.stdout.write(
                        f"\r{self._filename}: {percentage:.2f}% - {self._seen_so_far}/{self._size} octets "
                        f"({speed:.2f} KB/s)"
                    )
                else:
                    sys.stdout.write(
                        f"\r{self._filename}: {self._seen_so_far} octets ({speed:.2f} KB/s)"
                    )
            else:
                if self._size > 0:
                    sys.stdout.write(
                        f"\r{self._filename}: {percentage:.2f}% - {self._seen_so_far}/{self._size} octets"
                    )
                else:
                    sys.stdout.write(
                        f"\r{self._filename}: {self._seen_so_far} octets"
                    )
            sys.stdout.flush()

def create_status_file(base_name, status="NOT_DONE"):
    """
    Crée un fichier d'état pour l'audio traité.
    
    Args:
        base_name: Nom de base du fichier audio (sans extension)
        status: État à écrire dans le fichier (NOT_DONE ou DONE)
        
    Returns:
        str: Chemin du fichier d'état créé
    """
    status_file = f"{base_name}.txt"
    
    try:
        with open(status_file, 'w') as f:
            f.write(status)
        print(f"Fichier d'état créé: {status_file} avec statut: {status}")
        return status_file
    except Exception as e:
        print(f"Erreur lors de la création du fichier d'état {status_file}: {e}")
        send_discord_error("Erreur de fichier d'état", f"Erreur lors de la création du fichier d'état {status_file}: {e}", traceback.format_exc())
        return None

def update_status_file(status_file, status):
    """
    Met à jour l'état dans le fichier d'état.
    
    Args:
        status_file: Chemin du fichier d'état
        status: Nouvel état à écrire
        
    Returns:
        bool: True si la mise à jour a réussi, False sinon
    """
    if not status_file:
        return False
        
    try:
        with open(status_file, 'w') as f:
            f.write(status)
        print(f"Fichier d'état mis à jour: {status_file} avec statut: {status}")
        return True
    except Exception as e:
        print(f"Erreur lors de la mise à jour du fichier d'état {status_file}: {e}")
        send_discord_error("Erreur de mise à jour du fichier d'état", f"Erreur lors de la mise à jour du fichier d'état {status_file}: {e}", traceback.format_exc())
        return False

def upload_to_cloudflare(file_path, target_name):
    """
    Upload un fichier vers Cloudflare R2 et attendre que le téléchargement soit terminé.
    Retourne True si le téléchargement est réussi, False sinon.
    
    Args:
        file_path: Chemin du fichier local à télécharger
        target_name: Nom à donner au fichier sur Cloudflare R2
    """
    try:            
        # Récupération des informations d'identification depuis les variables d'environnement
        access_key_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_ACCESS_KEY_ID")
        account_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_ACCOUNT_ID")
        secret_access_key = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_VOLUME_RUNPOD_NAME")
        
        # Vérification que toutes les variables d'environnement nécessaires sont définies
        if not all([access_key_id, account_id, secret_access_key, bucket_name]):
            error_msg = "Variables d'environnement manquantes pour Cloudflare R2"
            print("ERREUR CRITIQUE: Variables d'environnement manquantes pour Cloudflare R2")
            print(f"  - ACCESS_KEY_ID présent: {bool(access_key_id)}")
            print(f"  - ACCOUNT_ID présent: {bool(account_id)}")
            print(f"  - SECRET_ACCESS_KEY présent: {bool(secret_access_key)}")
            print(f"  - BUCKET_NAME présent: {bool(bucket_name)}")
            send_discord_error("Configuration Cloudflare R2", error_msg)
            return False

        # Configuration du client S3 pour Cloudflare R2
        r2 = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )
        
        print(f"Début du téléchargement de {file_path} vers Cloudflare R2 (sera nommé '{target_name}')...")
        
        # Utilisation de upload_file qui gère automatiquement les téléchargements volumineux
        # et fournit une progression
        r2.upload_file(
            Filename=file_path,
            Bucket=bucket_name,
            Key=target_name,
            Callback=ProgressPercentage(file_path)
        )
        
        # Vérification que le fichier existe bien dans le bucket
        response = r2.head_object(Bucket=bucket_name, Key=target_name)
        print(f"Téléchargement terminé avec succès! Taille: {response['ContentLength']} octets")
        return True
    
    except Exception as e:
        print(f"ERREUR lors du téléchargement vers Cloudflare R2: {e}")
        traceback.print_exc()
        send_discord_error("Erreur de téléchargement Cloudflare", f"ERREUR lors du téléchargement vers Cloudflare R2: {e}", traceback.format_exc())
        return False

def download_from_url(url, output_file):
    """
    Télécharge un fichier depuis une URL et le sauvegarde localement.
    
    Args:
        url: URL du fichier à télécharger
        output_file: Chemin où sauvegarder le fichier
        
    Returns:
        bool: True si le téléchargement a réussi, False sinon
    """
    try:
        print(f"Téléchargement depuis l'URL: {url}")
        
        # Vérifier si le dossier de destination existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Télécharger le fichier avec une barre de progression
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lever une exception pour les erreurs HTTP
        
        # Obtenir la taille totale du fichier s'il est disponible
        total_size = int(response.headers.get('content-length', 0))
        
        # Télécharger et écrire le fichier par morceaux
        with open(output_file, 'wb') as f:
            if total_size == 0:
                # Si la taille est inconnue, télécharger sans barre de progression
                f.write(response.content)
            else:
                # Sinon, utiliser une barre de progression
                downloaded = 0
                last_time = time.time()
                last_downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filtrer les keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Afficher la progression
                        current_time = time.time()
                        if current_time - last_time >= 1.0:  # Mise à jour toutes les secondes
                            progress = (downloaded / total_size) * 100
                            speed = (downloaded - last_downloaded) / (current_time - last_time) / 1024  # KB/s
                            
                            sys.stdout.write(
                                f"\rTéléchargement: {progress:.2f}% - {downloaded}/{total_size} octets "
                                f"({speed:.2f} KB/s)"
                            )
                            sys.stdout.flush()
                            
                            last_time = current_time
                            last_downloaded = downloaded
        
        print(f"\nTéléchargement terminé: {output_file}")
        return True
        
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis {url}: {e}")
        send_discord_error("Erreur de téléchargement URL", f"Erreur lors du téléchargement depuis {url}: {e}", traceback.format_exc())
        
        # Si le fichier a été partiellement téléchargé, le supprimer
        if os.path.exists(output_file):
            os.remove(output_file)
            
        return False

def download_from_r2(r2_path, local_output_path):
    """
    Télécharge un fichier depuis Cloudflare R2 en utilisant les informations d'accès
    des variables d'environnement.
    
    Args:
        r2_path: Chemin du fichier dans le bucket R2
        local_output_path: Chemin local où sauvegarder le fichier
        
    Returns:
        bool: True si le téléchargement a réussi, False sinon
    """
    try:
        # Récupération des informations d'identification depuis les variables d'environnement
        access_key_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_ACCESS_KEY_ID")
        account_id = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_ACCOUNT_ID")
        secret_access_key = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get(f"{PROVIDER_POD}CLOUDFARE_R2_VOLUME_RUNPOD_NAME")
        
        # Vérification que toutes les variables d'environnement nécessaires sont définies
        if not all([access_key_id, account_id, secret_access_key, bucket_name]):
            error_msg = "Variables d'environnement manquantes pour Cloudflare R2"
            print("Erreur: Variables d'environnement manquantes pour Cloudflare R2")
            send_discord_error("Configuration Cloudflare R2", error_msg)
            return False
            
        print(f"Téléchargement depuis R2: {bucket_name}/{r2_path}")
        
        # Configuration du client S3 pour Cloudflare R2
        r2 = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )
        
        # Vérifier si le fichier existe dans le bucket
        try:
            r2.head_object(Bucket=bucket_name, Key=r2_path)
        except Exception as e:
            error_msg = f"Le fichier {r2_path} n'existe pas dans le bucket {bucket_name}: {e}"
            print(f"Erreur: {error_msg}")
            send_discord_error("Fichier R2 introuvable", error_msg, traceback.format_exc())
            return False
            
        # Télécharger le fichier
        print(f"Téléchargement de {r2_path} vers {local_output_path}...")
        
        # Créer le dossier de destination si nécessaire
        os.makedirs(os.path.dirname(local_output_path) if os.path.dirname(local_output_path) else '.', exist_ok=True)
        
        # Télécharger avec suivi de progression
        r2.download_file(
            Bucket=bucket_name,
            Key=r2_path,
            Filename=local_output_path,
            Callback=ProgressPercentage(local_output_path, is_download=True)
        )
        
        if os.path.exists(local_output_path):
            print(f"\nTéléchargement réussi: {local_output_path}")
            return True
        else:
            error_msg = f"Le téléchargement a échoué, le fichier {local_output_path} n'existe pas"
            print(f"Erreur: {error_msg}")
            send_discord_error("Téléchargement R2 échoué", error_msg)
            return False
            
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis R2: {e}")
        send_discord_error("Erreur de téléchargement R2", f"Erreur lors du téléchargement depuis R2: {e}", traceback.format_exc())
        return False

def is_url(s):
    """
    Vérifie si une chaîne de caractères est une URL.
    
    Args:
        s: Chaîne à vérifier
        
    Returns:
        bool: True si c'est une URL, False sinon
    """
    # Vérification simple pour les URLs HTTP/HTTPS
    return s.startswith('http://') or s.startswith('https://')

def get_audio_duration(filename):
    """Obtient la durée d'un fichier audio en secondes"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', filename]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Erreur lors de l'obtention de la durée audio: {e}")
        send_discord_error("Erreur d'analyse audio", f"Erreur lors de l'obtention de la durée du fichier audio {filename}: {e}", traceback.format_exc())
        raise

def split_audio_file(input_file, chunk_duration=600):  # 600 secondes = 10 minutes
    """Divise un fichier audio en morceaux de durée spécifiée en utilisant pydub"""
    
    # Récupérer la durée en utilisant pydub
    def get_audio_duration(file):
        audio = AudioSegment.from_file(file)
        return audio.duration_seconds
    
    duration = get_audio_duration(input_file)
    
    if duration <= chunk_duration:
        return [input_file]  # Pas besoin de découper
    
    # Calculer le nombre de morceaux nécessaires
    num_chunks = math.ceil(duration / chunk_duration)
    chunk_files = []
    
    # Charger l'audio avec pydub
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier audio: {e}")
        if 'send_discord_error' in globals():
            send_discord_error("Erreur de chargement audio", 
                              f"Erreur lors du chargement du fichier audio: {e}", 
                              traceback.format_exc())
        return [input_file]
    
    # Créer un dossier temporaire pour les morceaux
    temp_dir = tempfile.mkdtemp()
    
    # Convertir la durée en millisecondes pour pydub
    chunk_duration_ms = chunk_duration * 1000
    
    for i in range(num_chunks):
        start_time = i * chunk_duration_ms
        end_time = min((i + 1) * chunk_duration_ms, len(audio))
        
        output_file = os.path.join(temp_dir, f"part{i+1}.wav")
        
        try:
            # Extraire le segment
            chunk = audio[start_time:end_time]
            # Exporter au format WAV
            chunk.export(output_file, format="wav")
            chunk_files.append(output_file)
        except Exception as e:
            print(f"Erreur lors de la découpe du fichier: {e}")
            if 'send_discord_error' in globals():
                send_discord_error("Erreur de découpage audio", 
                                 f"Erreur lors de la découpe du fichier audio (segment {i+1}): {e}", 
                                 traceback.format_exc())
            
            # Continuer avec les fichiers déjà créés
            if not chunk_files:
                # Si aucun morceau n'a été créé, retourner le fichier d'origine
                return [input_file]
    
    return chunk_files

def get_gpu_memory_category():
    try:
        pynvml.nvmlInit()
        
        deviceCount = pynvml.nvmlDeviceGetCount()
        if deviceCount == 0:
            print("Aucun GPU NVIDIA détecté")
            return "",50000
            
        # On prend le premier GPU pour simplicité
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Conversion en gigaoctets
        total_memory_gb = info.total / (1024**3)
        
        print(f"Mémoire GPU totale détectée: {total_memory_gb:.2f} Go")
        
        # Application des critères
        if 0 <= total_memory_gb < 8:
            return "",50000
        elif 8 <= total_memory_gb < 13:
            return "",200000
        elif 13 <= total_memory_gb < 15:
            return "--large_gpu",250000
        elif 15 <= total_memory_gb < 20:
            return "--large_gpu",500000
        elif 20 <= total_memory_gb <= 24:
            return "--large_gpu",1000000
        else:
            return "--large_gpu",1500000
    
    except Exception as e:
        print(f"Erreur lors de la détection de la mémoire GPU: {e}")
        send_discord_error("Erreur de détection GPU", f"Erreur lors de la détection de la mémoire GPU: {e}", traceback.format_exc())
        return "",50000  # Valeur par défaut en cas d'erreur
    
    finally:
        pynvml.nvmlShutdown()

def monitor_output_generation(func):
    @functools.wraps(func)
    def wrapper(chunk_files, output_folder, args, *moreargs, **kwargs):
        # Capturer les fichiers existants avant le traitement
        existing_files = set()
        if os.path.exists(output_folder):
            existing_files = set(os.listdir(output_folder))
        
        # Valeurs pour le suivi de progression
        start_increment = 25
        pourcentage_avancement = 60/len(chunk_files)
        current_progress = start_increment+pourcentage_avancement
        
        # Nombre de paires de fichiers attendues
        processed_chunks = 0
        
        def check_new_files():
            nonlocal processed_chunks, current_progress, existing_files
            
            current_files = set(os.listdir(output_folder))
            new_files = current_files - existing_files
            
            new_pairs_count = len(new_files) // 2
            
            if new_pairs_count > processed_chunks:
                pairs_to_process = new_pairs_count - processed_chunks
                
                for _ in range(pairs_to_process):
                    processed_chunks += 1
                    push_kv_runpod(current_progress)
                    print(f"Progression: {current_progress:.2f}% - Traité {processed_chunks}/{len(chunk_files)} fichiers")
                    current_progress += pourcentage_avancement
                
                existing_files.update(new_files)
        
        stop_monitoring = False
        
        def monitor_folder():
            while not stop_monitoring:
                if os.path.exists(output_folder):
                    check_new_files()
                time.sleep(1)  # Vérifier toutes les 200ms
        
        monitor_thread = threading.Thread(target=monitor_folder)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            result = func(chunk_files, output_folder, args, *moreargs, **kwargs)
            time.sleep(1)  # Attente réduite
            check_new_files()
            return result
        finally:
            stop_monitoring = True
            monitor_thread.join(timeout=2)
    
    return wrapper

@monitor_output_generation
def process_files_with_inference(chunk_files, output_folder, args):
    """
    Traite tous les fichiers découpés avec le script d'inférence
    
    Returns:
        bool: True si le traitement a réussi, False sinon
    """
    # Récupérer le nom du script d'inférence du premier argument
    inference_script = args[0]
    
    try:
        # Construire les arguments pour inference.py
        input_args = ["--input_audio"] + chunk_files
        orig_args = args.copy()
        
        # Remplacer l'argument input_audio par notre liste de fichiers
        if "--input_audio" in orig_args:
            idx = orig_args.index("--input_audio")
            if idx + 1 < len(orig_args):
                orig_args = orig_args[:idx] + orig_args[idx+2:]
        
        # S'assurer que le dossier de sortie existe
        os.makedirs(output_folder, exist_ok=True)
        
        large, nb_chunk = get_gpu_memory_category()
        # Exécuter le script d'inférence avec les arguments
        full_args = [inference_script] + input_args + orig_args[1:] + [el for el in ["--output_folder",output_folder,large,"--only_vocals","--overlap_large","0.0001","--overlap_small","1","--chunk_size", str(nb_chunk)] if el!=""]
        sys.argv = full_args
        print(f"Exécution de {inference_script} avec les arguments: {' '.join(full_args)}")
        
        # Vérification de l'existence du fichier
        if not os.path.isfile(inference_script):
            error_msg = f"Le fichier {inference_script} n'existe pas"
            print(f"ERREUR CRITIQUE: {error_msg}")
            send_discord_error("Fichier d'inférence manquant", error_msg)
            return False

        runpy.run_path(inference_script, run_name='__main__')
        return True
    except FileNotFoundError as e:
        print(f"ERREUR CRITIQUE - Fichier non trouvé: {e}")
        traceback.print_exc()
        send_discord_error("Fichier non trouvé", f"ERREUR CRITIQUE - Fichier non trouvé: {e}", traceback.format_exc())
        return False
    except Exception as e:
        print(f"ERREUR CRITIQUE lors de l'exécution de {inference_script}: {e}")
        traceback.print_exc()
        send_discord_error("Erreur d'inférence", f"ERREUR CRITIQUE lors de l'exécution de {inference_script}: {e}", traceback.format_exc())
        return False

def concatenate_audio_files(files, output_file):
    """
    Concatène plusieurs fichiers audio en un seul
    
    Returns:
        bool: True si la concaténation a réussi, False sinon
    """
    try:
        # Créer un fichier de liste pour ffmpeg
        list_file = "concat_list.txt"
        with open(list_file, "w") as f:
            for file in files:
                f.write(f"file '{file}'\n")
        
        # Concaténer les fichiers
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_file]
        subprocess.run(cmd, check=True)
        
        # Supprimer le fichier de liste
        os.remove(list_file)
        return True
    except Exception as e:
        print(f"Erreur lors de la concaténation des fichiers: {e}")
        traceback.print_exc()
        send_discord_error("Erreur de concaténation", f"Erreur lors de la concaténation des fichiers audio: {e}", traceback.format_exc())
        return False

def convert_to_mono_flac(input_file, output_file):
    """
    Convertit un fichier audio en FLAC mono-canal
    
    Returns:
        bool: True si la conversion a réussi, False sinon
    """
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(input_file):
        error_msg = f"Le fichier d'entrée {input_file} n'existe pas"
        print(f"Erreur: {error_msg}")
        send_discord_error("Fichier introuvable", error_msg)
        return False
    
    # Si le fichier de sortie existe déjà, essayer de le supprimer d'abord
    if os.path.exists(output_file):
        try:
            print(f"Le fichier {output_file} existe déjà, tentative de suppression...")
            os.remove(output_file)
            print(f"Fichier {output_file} supprimé avec succès")
        except OSError as e:
            print(f"Impossible de supprimer le fichier existant {output_file}: {e}")
            # Continuer malgré l'erreur, ffmpeg avec -y devrait écraser le fichier
    
    # Conversion simple en FLAC mono
    cmd = ['ffmpeg', '-y', '-i', input_file, '-ac', '1', output_file]
    
    try:
        print(f"Conversion de {input_file} en {output_file}...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"Conversion réussie: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        
        # Si la première méthode échoue, essayer avec un codec explicite
        try:
            print("Tentative avec codec explicite...")
            alt_cmd = ['ffmpeg', '-y', '-i', input_file, '-ac', '1', '-c:a', 'flac', output_file]
            result = subprocess.run(alt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(f"Conversion alternative réussie: {output_file}")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"Échec de la conversion alternative: {e2}")
            print(f"Sortie d'erreur: {e2.stderr}")
            send_discord_error("Erreur de conversion audio", f"Échec de la conversion audio: {e2}", e2.stderr)
            return False

def cleanup(files):
    """Supprime une liste de fichiers"""
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Fichier supprimé: {file}")
            except Exception as e:
                print(f"Impossible de supprimer {file}: {e}")
                send_discord_error("Erreur de nettoyage", f"Impossible de supprimer le fichier {file}: {e}")

# Programme principal
if __name__ == "__main__":
    # Configurer un timer pour interrompre l'exécution si elle prend trop de temps
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_EXECUTION_TIME)
    
    # Statut global pour suivre si tout s'est bien passé
    global_success = False
    status_file = None
    status_file_r2_path = None
    
    try:
        # Le nom cible pour Cloudflare est le premier argument
        cloudflare_target_name = sys.argv[1]
        
        # Récupérer les arguments d'origine (sans cr.py et nom_cible.flac)
        original_args = sys.argv[2:]

        # Le nom cible pour Cloudflare est le premier argument
        cloudflare_target_name = sys.argv[1]
        
        # Récupérer les arguments d'origine (sans cr.py et nom_cible.flac)
        original_args = sys.argv[2:]
        
        # Analyser les arguments pour trouver --id_customer
        id_customer = None
        i = 0
        while i < len(original_args):
            if original_args[i] == "--id_customer":
                if i + 1 < len(original_args):
                    id_customer = original_args[i + 1]
                    # Supprimer l'argument et sa valeur de la liste des arguments originaux
                    original_args.pop(i)
                    original_args.pop(i)
                    break
                else:
                    print("Erreur: Valeur manquante pour --id_customer")
                    sys.exit(1)
            i += 1

        print("id customer = ",id_customer)
        genlab_customer_id= str(id_customer)
        # Trouver le fichier d'entrée et le dossier de sortie
        input_source = None
        input_arg_idx = -1
        output_folder = "./results/"
        
        i = 0
        while i < len(original_args):
            if original_args[i] == "--input_audio" and i+1 < len(original_args):
                input_source = original_args[i+1]
                input_arg_idx = i+1
                i += 2  # Sauter l'argument et sa valeur
            elif original_args[i] == "--output_folder" and i+1 < len(original_args):
                output_folder = original_args[i+1]
                i += 2  # Sauter l'argument et sa valeur
            else:
                i += 1  # Avancer au prochain argument

        if not input_source:
            error_msg = "Aucun fichier d'entrée spécifié"
            print(error_msg)
            send_discord_error("Entrée manquante", error_msg)
            sys.exit(1)
        
        print(f"Source d'entrée: {input_source}")
        
        # Créer le chemin du fichier d'état R2
        # Le fichier d'état aura toujours le même nom que le fichier d'entrée mais avec extension .txt
        if is_url(input_source):
            # Pour les URLs, extraire le nom du fichier de l'URL
            url_filename = os.path.basename(input_source.split('?')[0])  # Enlever les paramètres de l'URL
            base_name = os.path.splitext(url_filename)[0]
            status_file_r2_path = base_name + '.txt'
        else:
            # Pour les chemins R2 ou fichiers locaux, extraire juste le nom du fichier
            input_file_name = os.path.basename(input_source)
            base_name = os.path.splitext(input_file_name)[0]
            status_file_r2_path = base_name + '.txt'
            
        print(f"Fichier d'état qui sera téléversé: {status_file_r2_path}")
        
        # Créer le fichier d'état local (NOT_DONE par défaut)
        status_file = create_status_file(base_name, "NOT_DONE")
        
        # Vérifier le type de source
        if is_url(input_source):
            # C'est une URL
            input_file = "vid.flac"  # Nom par défaut pour le fichier téléchargé
            print(f"URL détectée, téléchargement vers {input_file}...")
            
            # Télécharger l'URL vers un fichier local
            if not download_from_url(input_source, input_file):
                error_msg = f"Échec du téléchargement depuis {input_source}"
                print(error_msg)
                send_discord_error("Téléchargement échoué", error_msg)
                sys.exit(1)
                
            # Remplacer l'URL par le chemin local dans les arguments
            original_args[input_arg_idx] = input_file
            
            # Marquer que c'est un fichier téléchargé à supprimer plus tard
            is_temp_file = True
            
        elif os.path.isfile(input_source):
            # C'est un fichier local
            input_file = input_source
            print(f"Fichier local trouvé: {input_file}")
            
            # Ce n'est pas un fichier temporaire
            is_temp_file = False
            
        else:
            # Ce n'est ni une URL ni un fichier local, on suppose que c'est un chemin dans le bucket R2
            print(f"Fichier local non trouvé, tentative de téléchargement depuis Cloudflare R2...")
            input_file = os.path.basename(input_source)  # Utiliser juste le nom du fichier comme destination locale
            
            # Télécharger depuis Cloudflare R2
            if not download_from_r2(input_source, input_file):
                error_msg = f"Échec du téléchargement depuis Cloudflare R2: {input_source}"
                print(error_msg)
                send_discord_error("Téléchargement R2 échoué", error_msg)
                sys.exit(1)
            else:
                # premiere mise a jour de telchargement du fichier
                push_kv_runpod(20)
                
            # Remplacer le chemin R2 par le chemin local dans les arguments
            original_args[input_arg_idx] = input_file
            
            # Marquer que c'est un fichier téléchargé à supprimer plus tard
            is_temp_file = True
        
        # Découper le fichier si nécessaire
        chunk_files = split_audio_file(input_file)
        # deuxieme mise a jour de split du fichier audio
        push_kv_runpod(25)
        
        # Traiter les morceaux
        if not process_files_with_inference(chunk_files, output_folder, original_args):
            error_msg = "Erreur lors du traitement des fichiers avec le script d'inférence"
            print(error_msg)
            send_discord_error("Erreur de traitement", error_msg)
            raise Exception(error_msg)
        
        # Rassembler les fichiers instrum
        current_dir = os.getcwd()
        os.chdir(output_folder)
        instrum_files = sorted(glob.glob("*_instrum.wav"))
        
        output_flac = "full_instrumental.flac"
        conversion_success = False
        
        if len(instrum_files) > 1:
            # Concaténer tous les fichiers instrum
            temp_concat = "temp_concat.wav"
            if not concatenate_audio_files(instrum_files, temp_concat):
                error_msg = "Erreur lors de la concaténation des fichiers instrum"
                print(error_msg)
                send_discord_error("Erreur de concaténation", error_msg)
                raise Exception(error_msg)
            
            # Convertir en mono FLAC
            conversion_success = convert_to_mono_flac(temp_concat, output_flac)
            
            # Nettoyer
            cleanup([temp_concat] + instrum_files + glob.glob("*_vocals.wav"))
        elif len(instrum_files) == 1:
            # Juste convertir en mono FLAC
            conversion_success = convert_to_mono_flac(instrum_files[0], output_flac)
            cleanup(instrum_files + glob.glob("*_vocals.wav"))
        else:
            error_msg = "Aucun fichier instrumental trouvé!"
            print(error_msg)
            send_discord_error("Fichiers manquants", error_msg)
            raise Exception(error_msg)
        
        if not conversion_success:
            error_msg = "Erreur lors de la conversion en FLAC mono"
            print(error_msg)
            send_discord_error("Erreur de conversion", error_msg)
            raise Exception(error_msg)
        
        # Vérifier que le fichier instrumental final existe dans le dossier de sortie
        final_path = os.path.join(os.getcwd(), output_flac)
        
        # quatrieme mise a jour
        push_kv_runpod(85)
        if os.path.exists(final_path):
            print(f"Fichier final trouvé: {final_path}")
            
            # Téléverser le fichier instrumental final vers Cloudflare R2
            upload_success = upload_to_cloudflare(final_path, cloudflare_target_name)
            
            if upload_success:
                print(f"Le fichier {final_path} a été téléversé avec succès vers Cloudflare R2 sous le nom {cloudflare_target_name}")
                global_success = True
            else:
                error_msg = f"Échec du téléversement du fichier {final_path} vers Cloudflare R2"
                print(error_msg)
                send_discord_error("Échec du téléversement", error_msg)
                raise Exception(error_msg)
        else:
            error_msg = f"Erreur: Le fichier final {final_path} n'existe pas"
            print(error_msg)
            send_discord_error("Fichier final manquant", error_msg)
            
            # Chercher d'autres fichiers FLAC qui pourraient être téléversés
            other_flacs = glob.glob("*.flac")
            if other_flacs:
                print(f"Autres fichiers FLAC trouvés: {other_flacs}")
                
                # Tenter de téléverser le premier fichier FLAC trouvé
                alternative_path = os.path.join(os.getcwd(), other_flacs[0])
                print(f"Tentative de téléversement du fichier alternatif: {alternative_path}")
                
                upload_success = upload_to_cloudflare(alternative_path, cloudflare_target_name)
                
                if upload_success:
                    print(f"Le fichier alternatif {alternative_path} a été téléversé avec succès vers Cloudflare R2 sous le nom {cloudflare_target_name}")
                    global_success = True
                else:
                    error_msg = f"Échec du téléversement du fichier alternatif {alternative_path} vers Cloudflare R2"
                    print(error_msg)
                    send_discord_error("Échec du téléversement alternatif", error_msg)
                    raise Exception(error_msg)
            else:
                error_msg = "Aucun fichier FLAC trouvé à téléverser"
                print(error_msg)
                send_discord_error("Aucun fichier à téléverser", error_msg)
                raise Exception(error_msg)
        
        # Retour au répertoire de travail initial
        os.chdir(current_dir)
        
        # Supprimer les fichiers de morceaux temporaires si on en a créé
        if len(chunk_files) > 1 and os.path.exists("temp_chunks"):
            for file in chunk_files:
                if os.path.exists(file):
                    os.remove(file)
            os.rmdir("temp_chunks")
        
        # Supprimer le fichier téléchargé si c'était un fichier temporaire
        if is_temp_file and os.path.exists(input_file):
            try:
                os.remove(input_file)
                os.remove(os.path.join(output_folder, output_flac))
                print(f"Fichier temporaire supprimé: {input_file}")
            except Exception as e:
                print(f"Impossible de supprimer le fichier temporaire {input_file}: {e}")
        
        print(f"Traitement terminé avec succès. Résultat final: {os.path.join(output_folder, output_flac)}")
        
        # Désactiver l'alarme car le script s'est terminé avec succès
        signal.alarm(0)
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        traceback.print_exc()
        send_discord_error("Erreur de traitement général", f"Erreur lors du traitement: {e}", traceback.format_exc())
        global_success = False
    
    finally:
        # Désactiver l'alarme dans tous les cas
        signal.alarm(0)
        
        # Mettre à jour le fichier d'état en fonction du résultat global
        try:
            print("\n--- MISE À JOUR DU STATUT FINAL ---")
            final_status = "DONE" if global_success else "NOT_DONE"
            print(f"Statut final: {final_status}")
            
            # Si status_file n'a pas été défini, créer un fichier avec un nom par défaut
            if not status_file:
                print("Aucun fichier d'état n'a été créé précédemment, création d'un fichier par défaut")
                status_file = create_status_file(base_name, final_status)
                
                # Si status_file_r2_path n'a pas été défini, créer un chemin par défaut
                if not status_file_r2_path:
                    status_file_r2_path = f"{base_name}.txt"
                    print(f"Aucun chemin R2 n'a été défini pour le fichier d'état, utilisation de {status_file_r2_path}")
            else:
                # Mise à jour du fichier d'état existant
                print(f"Mise à jour du fichier d'état: {status_file} → {final_status}")
                update_success = update_status_file(status_file, final_status)
                
                if not update_success:
                    print(f"AVERTISSEMENT: Échec de mise à jour du fichier d'état local {status_file}")
                    # Tentative de recréation si la mise à jour échoue
                    print(f"Tentative de recréation du fichier d'état...")
                    status_file = create_status_file(base_name, final_status)
            
            # Téléversement du fichier d'état
            if status_file and status_file_r2_path:
                print(f"Tentative de téléversement du fichier d'état {status_file} vers {status_file_r2_path}...")
                
                # Vérification que le fichier d'état existe
                if not os.path.exists(status_file):
                    print(f"ERREUR: Le fichier d'état {status_file} n'existe pas! Tentative de le recréer...")
                    status_file = create_status_file(base_name, final_status)
                    
                    # Vérifier à nouveau après la tentative de création
                    if not os.path.exists(status_file):
                        print(f"ERREUR CRITIQUE: Impossible de créer le fichier d'état {status_file}")
                        print("Tentative de création d'un fichier d'état alternatif...")
                        alt_status_file = "status_fallback.txt"
                        with open(alt_status_file, "w") as f:
                            f.write(final_status)
                        status_file = alt_status_file
                
                # Téléversement avec tentatives multiples
                upload_attempts = 0
                upload_success = False
                
                while upload_attempts < 3 and not upload_success:
                    upload_attempts += 1
                    print(f"Tentative {upload_attempts}/3 de téléversement de {status_file}...")
                    
                    try:
                        # Vérifier à nouveau l'existence du fichier avant chaque tentative
                        if os.path.exists(status_file):
                            upload_success = upload_to_cloudflare(status_file, status_file_r2_path)
                        else:
                            print(f"ERREUR: Le fichier {status_file} n'existe pas pour le téléversement")
                            break
                            
                        if upload_success:
                            print(f"Fichier d'état téléversé avec succès vers Cloudflare R2: {status_file_r2_path}")
                        else:
                            print(f"Échec du téléversement à la tentative {upload_attempts}/3")
                            if upload_attempts < 3:
                                print("Nouvelle tentative dans 2 secondes...")
                                time.sleep(2)
                    except Exception as upload_err:
                        print(f"ERREUR lors de la tentative {upload_attempts}: {upload_err}")
                        if upload_attempts < 3:
                            print("Nouvelle tentative après erreur dans 3 secondes...")
                            time.sleep(3)
                
                if not upload_success:
                    print("ERREUR CRITIQUE: Impossible de téléverser le fichier d'état après plusieurs tentatives")
                    send_discord_error("Échec du téléversement du statut", "Impossible de téléverser le fichier d'état après plusieurs tentatives")
                    
                    # Dernière tentative - créer un fichier avec un contenu minimal et essayer de le téléverser
                    print("Tentative de dernier recours avec un fichier minimal...")
                    emergency_file = "emergency_status.txt"
                    try:
                        with open(emergency_file, "w") as f:
                            f.write(final_status)
                        
                        if upload_to_cloudflare(emergency_file, status_file_r2_path):
                            print("Téléversement d'urgence réussi!")
                        else:
                            print("Échec du téléversement d'urgence")
                            send_discord_error("Échec du téléversement d'urgence", f"Impossible de téléverser le fichier d'état d'urgence {emergency_file}")
                    except Exception as last_err:
                        print(f"Échec de la dernière tentative: {last_err}")
                        send_discord_error("Échec ultime du téléversement", f"Échec de la dernière tentative de téléversement: {last_err}")
            else:
                print("AVERTISSEMENT: Impossible de téléverser le fichier d'état car status_file ou status_file_r2_path n'est pas défini")
                print(f"  - status_file: {status_file}")
                print(f"  - status_file_r2_path: {status_file_r2_path}")
                send_discord_error("Fichier d'état indéfini", f"Impossible de téléverser le fichier d'état car status_file ou status_file_r2_path n'est pas défini. status_file: {status_file}, status_file_r2_path: {status_file_r2_path}")
                
        except Exception as e:
            print(f"ERREUR CRITIQUE lors de la mise à jour du fichier d'état: {e}")
            traceback.print_exc()
            send_discord_error("Erreur fatale de mise à jour du statut", f"ERREUR CRITIQUE lors de la mise à jour du fichier d'état: {e}", traceback.format_exc())

            # quatrieme mise a jour
            push_kv_runpod(100)  
        except Exception as e:
            id_machine = os.environ.get('RUNPOD_POD_ID') if PROVIDER_POD=="RUNPOD_SECRET_" or PROVIDER_POD == "" else os.environ.get('CONTAINER_ID')
            print(f"Échec de la tentative de de suppression du pod {id_machine}")
            send_discord_error("Échec de suppresion du pod ", f"Échec de la tentative de suppression du pod {PROVIDER_POD}: {id_machine}")
                    
        # Sortir avec le code d'erreur approprié
        print(f"\nFin du script avec statut: {'SUCCÈS' if global_success else 'ÉCHEC'}")
            
        sys.exit(0 if global_success else 1)
