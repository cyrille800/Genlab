#!/usr/bin/env python3
import torch
import importlib
from demucs.htdemucs import HTDemucs
import torch.serialization
import sys
import runpy
import os
import glob
import subprocess
import math
import tempfile
import boto3
import threading
import time
import requests
import datetime
import traceback
import signal

# Constante pour la durée maximale d'exécution (en secondes)
MAX_EXECUTION_TIME = 3600  # 1 heure par défaut, ajustez selon vos besoins

def send_discord_error(error_title, error_details, error_traceback=None):
    """
    Envoie un message d'erreur à un webhook Discord.
    
    Args:
        error_title: Titre de l'erreur
        error_details: Détails de l'erreur
        error_traceback: Traceback de l'erreur (optionnel)
    """
    try:
        webhook_url = os.environ.get("RUNPOD_SECRET_DISCORD_WEBHOOK_URL_SALON_ERROR")
        
        if not webhook_url:
            print("Erreur: URL du webhook Discord manquante dans les variables d'environnement")
            return False
            
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
        access_key_id = os.environ.get("RUNPOD_SECRET_CLOUDFARE_R2_ACCESS_KEY_ID")
        account_id = os.environ.get("RUNPOD_SECRET_CLOUDFARE_R2_ACCOUNT_ID")
        secret_access_key = os.environ.get("RUNPOD_SECRET_CLOUDFARE_R2_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get("RUNPOD_SECRET_CLOUDFARE_R2_VOLUME_RUNPOD_NAME")
        
        # Vérification que toutes les variables d'environnement nécessaires sont définies
        if not all([access_key_id, account_id, secret_access_key, bucket_name]):
            print("Erreur: Variables d'environnement manquantes pour Cloudflare R2")
            send_discord_error("Configuration Cloudflare R2", "Variables d'environnement manquantes pour Cloudflare R2")
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
        print(f"Erreur lors du téléchargement vers Cloudflare R2: {e}")
        send_discord_error("Erreur de téléchargement Cloudflare", f"Erreur lors du téléchargement vers Cloudflare R2: {e}", traceback.format_exc())
        return False

class ProgressPercentage:
    """Classe pour afficher la progression du téléchargement."""
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._last_time = time.time()
        self._last_seen = 0
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            
            # Calcul de la vitesse
            current_time = time.time()
            elapsed = current_time - self._last_time
            if elapsed >= 1.0:  # Afficher la vitesse toutes les secondes
                speed = (self._seen_so_far - self._last_seen) / elapsed / 1024  # KB/s
                self._last_time = current_time
                self._last_seen = self._seen_so_far
                sys.stdout.write(
                    f"\r{self._filename}: {percentage:.2f}% - {self._seen_so_far}/{self._size} octets "
                    f"({speed:.2f} KB/s)"
                )
            else:
                sys.stdout.write(
                    f"\r{self._filename}: {percentage:.2f}% - {self._seen_so_far}/{self._size} octets"
                )
            sys.stdout.flush()

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
        print(f"Erreur lors de l'obtention de la durée du fichier audio: {e}")
        send_discord_error("Erreur d'analyse audio", f"Erreur lors de l'obtention de la durée du fichier audio {filename}: {e}", traceback.format_exc())
        raise

def split_audio_file(input_file, chunk_duration=600):  # 600 secondes = 10 minutes
    """Divise un fichier audio en morceaux de durée spécifiée"""
    try:
        duration = get_audio_duration(input_file)
        
        if duration <= chunk_duration:
            return [input_file]  # Pas besoin de découper
        
        # Calculer le nombre de morceaux nécessaires
        num_chunks = math.ceil(duration / chunk_duration)
        chunk_files = []
        
        # Créer un dossier temporaire pour les morceaux
        temp_dir = "temp_chunks"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_file = os.path.join(temp_dir, f"part{i+1}.wav")
            
            # Utiliser ffmpeg pour extraire le segment
            if i == num_chunks - 1:  # Dernier morceau jusqu'à la fin
                cmd = ['ffmpeg', '-y', '-i', input_file, '-ss', str(start_time), 
                    '-acodec', 'pcm_s16le', output_file]
            else:
                cmd = ['ffmpeg', '-y', '-i', input_file, '-ss', str(start_time), 
                    '-t', str(chunk_duration), '-acodec', 'pcm_s16le', output_file]
            
            subprocess.run(cmd, check=True)
            chunk_files.append(output_file)
        
        return chunk_files
    except Exception as e:
        print(f"Erreur lors du découpage du fichier audio: {e}")
        send_discord_error("Erreur de découpage audio", f"Erreur lors du découpage du fichier audio {input_file}: {e}", traceback.format_exc())
        raise

def process_files_with_inference(chunk_files, output_folder, args):
    """Traite tous les fichiers découpés avec inference_modifier.py"""
    try:
        # Construire les arguments pour inference_modifier.py
        input_args = ["--input_audio"] + chunk_files
        orig_args = args.copy()
        
        # Remplacer l'argument input_audio par notre liste de fichiers
        if "--input_audio" in orig_args:
            idx = orig_args.index("--input_audio")
            if idx + 1 < len(orig_args):
                orig_args = orig_args[:idx] + orig_args[idx+2:]
        
        # S'assurer que le dossier de sortie existe
        os.makedirs(output_folder, exist_ok=True)
        
        # Exécuter inference_modifier.py avec les arguments
        full_args = ["inference_modifier.py"] + input_args + orig_args
        sys.argv = full_args
        runpy.run_path("inference_modifier.py", run_name='__main__')
    except Exception as e:
        print(f"Erreur lors du traitement avec inference_modifier.py: {e}")
        send_discord_error("Erreur de traitement", f"Erreur lors du traitement avec inference_modifier.py: {e}", traceback.format_exc())
        raise

def concatenate_audio_files(files, output_file):
    """Concatène plusieurs fichiers audio en un seul"""
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
    except Exception as e:
        print(f"Erreur lors de la concaténation des fichiers audio: {e}")
        send_discord_error("Erreur de concaténation", f"Erreur lors de la concaténation des fichiers audio: {e}", traceback.format_exc())
        raise

def convert_to_mono_flac(input_file, output_file):
    """Convertit un fichier audio en FLAC mono-canal"""
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
            os.remove(file)

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

# Programme principal
if __name__ == "__main__":
    # Configurer un timer pour interrompre l'exécution si elle prend trop de temps
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_EXECUTION_TIME)
    
    try:
        # Vérifier si on a au moins 3 arguments: cr.py, nom_cible.flac, inference_modifier.py
        if len(sys.argv) < 3:
            error_msg = "Usage: python3 cr.py <nom_target.flac> inference_modifier.py --input_audio <fichier_source_ou_url.flac> [autres arguments...]"
            print(error_msg)
            send_discord_error("Arguments insuffisants", error_msg)
            sys.exit(1)
        
        # Le nom cible pour Cloudflare est le premier argument
        cloudflare_target_name = sys.argv[1]
        
        # Récupérer les arguments d'origine (sans cr.py et nom_cible.flac)
        original_args = sys.argv[2:]
        
        # Trouver le fichier d'entrée et le dossier de sortie
        input_source = None
        input_arg_idx = -1
        output_folder = "./results/"
        
        # Chercher l'argument --input_audio et le fichier d'entrée
        if "--input_audio" in original_args:
            input_arg_idx = original_args.index("--input_audio")
            if input_arg_idx + 1 < len(original_args):
                input_source = original_args[input_arg_idx + 1]
        
        # Chercher l'argument --output_dir et le dossier de sortie
        if "--output_dir" in original_args:
            output_idx = original_args.index("--output_dir")
            if output_idx + 1 < len(original_args):
                output_folder = original_args[output_idx + 1]
        
        # Si on n'a pas trouvé de fichier d'entrée, erreur
        if not input_source:
            error_msg = "Aucun fichier d'entrée spécifié avec --input_audio"
            print(f"Erreur: {error_msg}")
            send_discord_error("Entrée manquante", error_msg)
            sys.exit(1)
            
        # Télécharger l'URL si c'est une URL
        input_file = input_source
        if is_url(input_source):
            # Créer un nom de fichier temporaire pour l'URL
            url_filename = os.path.basename(input_source.split('?')[0])  # Ignorer les paramètres de l'URL
            if not url_filename or '.' not in url_filename:
                url_filename = "downloaded_audio.flac"  # Nom par défaut
            input_file = os.path.join(tempfile.gettempdir(), url_filename)
            
            # Télécharger l'URL
            if not download_from_url(input_source, input_file):
                error_msg = f"Échec du téléchargement de l'URL: {input_source}"
                print(f"Erreur: {error_msg}")
                send_discord_error("Téléchargement échoué", error_msg)
                sys.exit(1)
        
        # Vérifier si le fichier d'entrée existe
        if not os.path.exists(input_file):
            error_msg = f"Le fichier d'entrée n'existe pas: {input_file}"
            print(f"Erreur: {error_msg}")
            send_discord_error("Fichier introuvable", error_msg)
            sys.exit(1)
        
        # Découper le fichier si nécessaire
        chunk_files = split_audio_file(input_file)
        
        # Traiter les fichiers découpés
        process_files_with_inference(chunk_files, output_folder, original_args)
        
        # Trouver les fichiers de sortie générés
        output_files = glob.glob(os.path.join(output_folder, "*.wav"))
        if not output_files:
            error_msg = f"Aucun fichier de sortie trouvé dans {output_folder}"
            print(f"Erreur: {error_msg}")
            send_discord_error("Résultat manquant", error_msg)
            sys.exit(1)
        
        # Si on a plusieurs fichiers, les concaténer
        final_output = os.path.join(output_folder, "final_output.wav")
        if len(output_files) > 1:
            # Trier les fichiers par ordre numérique
            output_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]) if os.path.basename(x).split('_')[0].isdigit() else 0)
            concatenate_audio_files(output_files, final_output)
        else:
            final_output = output_files[0]
        
        # Convertir en FLAC mono
        flac_output = os.path.join(output_folder, cloudflare_target_name)
        if not convert_to_mono_flac(final_output, flac_output):
            error_msg = f"Échec de la conversion en FLAC: {final_output}"
            print(f"Erreur: {error_msg}")
            send_discord_error("Conversion échouée", error_msg)
            sys.exit(1)
        
        # Télécharger le fichier vers Cloudflare
        if not upload_to_cloudflare(flac_output, cloudflare_target_name):
            error_msg = f"Échec du téléchargement vers Cloudflare: {flac_output}"
            print(f"Erreur: {error_msg}")
            send_discord_error("Téléchargement Cloudflare échoué", error_msg)
            sys.exit(1)
        
        # Nettoyer les fichiers temporaires
        cleanup(chunk_files)
        if len(output_files) > 1:
            cleanup(output_files)
            cleanup([final_output])
        
        # Si le fichier d'entrée était téléchargé, le supprimer aussi
        if is_url(input_source) and input_file != input_source:
            cleanup([input_file])
        
        print(f"Traitement terminé avec succès: {cloudflare_target_name}")
        
        # Désactiver l'alarme
        signal.alarm(0)
        
    except Exception as e:
        error_msg = f"Erreur générale: {e}"
        print(error_msg)
        send_discord_error("Erreur générale", error_msg, traceback.format_exc())
        sys.exit(1)
