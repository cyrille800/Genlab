import os
import runpod
import subprocess
import requests
import datetime
import traceback
import asyncio
import torch
import logging
import random
import time
from typing import Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Définir la variable PROVIDER_POD pour les variables d'environnement Discord
PROVIDER_POD = "RUNPOD_SECRET_"

# Détection du nombre de GPUs disponibles
num_gpus = torch.cuda.device_count()
logger.info(f"Worker initialisé avec {num_gpus} GPUs disponibles")

# Variables pour la gestion de la concurrence
request_rate = 0
gpu_load = {}

# Sémaphore pour limiter l'accès aux GPUs
gpu_semaphore = asyncio.Semaphore(num_gpus)
gpu_status = {i: False for i in range(num_gpus)}
gpu_lock = asyncio.Lock()

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
        
        logger.info(f"Message d'erreur envoyé à Discord: {error_title}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message à Discord: {e}")
        return False

def download_scripts():
    """Télécharge les scripts dans le répertoire spécifié"""
    script_dir = "/MVSEP-MDX23-music-separation-model"
    
    try:
        # S'assurer que les répertoires existent
        os.makedirs(script_dir, exist_ok=True)
        os.makedirs("/uploadfileuser", exist_ok=True)
        
        # Téléchargement des scripts dans le répertoire spécifique
        subprocess.run(f"curl -k https://raw.githubusercontent.com/cyrille800/Genlab/main/cr.py -o {script_dir}/cr.py", 
                      shell=True, check=True)
        subprocess.run(f"curl -k https://raw.githubusercontent.com/cyrille800/Genlab/main/inference_demucs.py -o {script_dir}/inference_demucs.py", 
                      shell=True, check=True)
        
        logger.info(f"Scripts téléchargés avec succès dans {script_dir}")
        return True
    except Exception as e:
        error_msg = f"Erreur lors du téléchargement des scripts: {e}"
        logger.error(error_msg)
        send_discord_error(
            "Erreur d'initialisation Serverless", 
            error_msg,
            traceback.format_exc()
        )
        return False

def init():
    """Initialisation - exécutée une seule fois au démarrage du worker"""
    if not download_scripts():
        return {"error": "Échec du téléchargement des scripts"}
    
    try:
        logger.info(f"Initialisation réussie. {num_gpus} GPUs détectés et disponibles.")
        return {"status": "success", "initialized": True, "gpu_count": num_gpus}
    except Exception as e:
        error_msg = f"Erreur lors de l'initialisation: {e}"
        send_discord_error(
            "Erreur d'initialisation Serverless", 
            error_msg,
            traceback.format_exc()
        )
        return {"status": "error", "error": error_msg}

def update_request_rate():
    """
    Simule des changements dans le taux de requêtes pour imiter des scénarios réels.
    À remplacer par une véritable métrique en production.
    """
    global request_rate
    request_rate = random.randint(20, 100)
    logger.debug(f"Taux de requêtes mis à jour: {request_rate}")

async def get_available_gpu():
    """Attend qu'un GPU soit disponible et le réserve"""
    await gpu_semaphore.acquire()
    async with gpu_lock:
        for gpu_id in range(num_gpus):
            if not gpu_status[gpu_id]:
                gpu_status[gpu_id] = True
                gpu_load[gpu_id] = time.time()
                logger.info(f"GPU {gpu_id} attribué")
                return gpu_id
    # Ne devrait jamais atteindre ce point si le sémaphore fonctionne correctement
    logger.error("Erreur inattendue dans l'attribution de GPU")
    return 0

async def release_gpu(gpu_id):
    """Libère un GPU après utilisation"""
    if gpu_id is not None and 0 <= gpu_id < num_gpus:
        async with gpu_lock:
            gpu_status[gpu_id] = False
            # Calculer le temps d'utilisation pour les métriques
            if gpu_id in gpu_load:
                usage_time = time.time() - gpu_load[gpu_id]
                logger.info(f"GPU {gpu_id} libéré après {usage_time:.2f} secondes")
                del gpu_load[gpu_id]
        gpu_semaphore.release()

async def process_request(event):
    """Traite les requêtes entrantes selon le format spécifié, avec gestion des GPUs"""
    try:
        # Récupérer les données d'entrée
        input_data = event.get("input", {})
        job_id = event.get("id", "unknown")
        
        # Récupérer les paramètres directement depuis l'entrée
        output_filename = input_data.get("output_filename")
        input_path = input_data.get("input_path")
        customer_id = input_data.get("customer_id", "123")
        
        # Vérifier que les paramètres requis sont présents
        if not output_filename or not input_path:
            error_msg = "Paramètres manquants. Veuillez fournir 'output_filename' et 'input_path'"
            send_discord_error(
                "Erreur de paramètres Serverless", 
                error_msg
            )
            return {"error": error_msg}

        # Obtenir un GPU disponible
        logger.info(f"Job {job_id}: en attente d'un GPU disponible")
        gpu_id = await get_available_gpu()
        logger.info(f"Job {job_id}: traitement sur GPU {gpu_id}")
        
        try:
            # Construire et exécuter la commande d'inférence avec le GPU attribué
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} python3 /MVSEP-MDX23-music-separation-model/cr.py {output_filename} "
                f"/MVSEP-MDX23-music-separation-model/inference_demucs.py "
                f"--input_audio {input_path} --id_customer {customer_id} --gpu_id 0"
            )
            
            logger.info(f"Job {job_id}: Exécution de la commande: {cmd}")
            start_time = time.time()
            
            # Utiliser asyncio.subprocess pour le traitement asynchrone
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            stdout, stderr = await process.communicate()
            stdout = stdout.decode()
            stderr = stderr.decode()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Job {job_id}: Traitement terminé en {elapsed_time:.2f} secondes")
            
            # Vérifier si le processus s'est bien terminé
            if process.returncode != 0:
                error_msg = f"Erreur pendant l'inférence avec code {process.returncode}"
                send_discord_error(
                    "Erreur d'inférence", 
                    f"Job {job_id} sur GPU {gpu_id}\nCommande: {cmd}\n\nErreur: {stderr}",
                    stderr
                )
                return {
                    "error": error_msg,
                    "command": cmd,
                    "stderr": stderr,
                    "gpu_used": gpu_id
                }
            
            return {
                "output": {
                    "success": True,
                    "message": "Traitement audio terminé",
                    "output_filename": output_filename,
                    "stdout": stdout,
                    "stderr": stderr,
                    "gpu_used": gpu_id,
                    "processing_time": f"{elapsed_time:.2f} secondes"
                }
            }
                
        except Exception as e:
            error_msg = f"Erreur pendant le traitement sur GPU {gpu_id}: {e}"
            send_discord_error(
                "Erreur de traitement", 
                error_msg,
                traceback.format_exc()
            )
            return {"error": error_msg, "gpu_used": gpu_id}
        finally:
            # Libérer le GPU quoi qu'il arrive
            await release_gpu(gpu_id)
            logger.info(f"Job {job_id}: GPU {gpu_id} libéré")
    
    except Exception as e:
        error_msg = f"Erreur inattendue pendant le traitement: {e}"
        send_discord_error(
            "Erreur inattendue Serverless", 
            error_msg,
            traceback.format_exc()
        )
        return {"error": error_msg}

def adjust_concurrency(current_concurrency):
    """
    Ajuste dynamiquement le niveau de concurrence basé sur le taux de requêtes observé.
    Suit le modèle de la documentation fournie, adapté au nombre de GPUs.
    """
    global request_rate
    update_request_rate()  # Simuler des changements dans le taux de requêtes

    max_concurrency = num_gpus  # Limite maximale basée sur le nombre de GPUs
    min_concurrency = 1  # Concurrence minimale à maintenir
    high_request_rate_threshold = 50  # Seuil pour un volume de requêtes élevé

    # Augmenter la concurrence si sous la limite max et taux de requêtes élevé
    if (
        request_rate > high_request_rate_threshold
        and current_concurrency < max_concurrency
    ):
        logger.info(f"Augmentation de la concurrence de {current_concurrency} à {current_concurrency + 1}")
        return current_concurrency + 1
    # Diminuer la concurrence si au-dessus de la limite min et taux de requêtes faible
    elif (
        request_rate <= high_request_rate_threshold
        and current_concurrency > min_concurrency
    ):
        logger.info(f"Réduction de la concurrence de {current_concurrency} à {current_concurrency - 1}")
        return current_concurrency - 1

    return current_concurrency

# Démarrer le service serverless avec gestion de la concurrence
runpod.serverless.start({
    "handler": process_request,
    "init": init,
    "concurrency_modifier": adjust_concurrency
})
