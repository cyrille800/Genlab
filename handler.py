import os
import runpod
import subprocess
import requests
import datetime
import traceback

# Définir la variable PROVIDER_POD pour les variables d'environnement Discord
PROVIDER_POD = "RUNPOD_SECRET_"

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
        
        print("Scripts téléchargés avec succès dans", script_dir)
        return True
    except Exception as e:
        error_msg = f"Erreur lors du téléchargement des scripts: {e}"
        print(error_msg)
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
        return {"status": "success", "initialized": True}
    except Exception as e:
        error_msg = f"Erreur lors de l'initialisation: {e}"
        send_discord_error(
            "Erreur d'initialisation Serverless", 
            error_msg,
            traceback.format_exc()
        )
        return {"status": "error", "error": error_msg}

def handler(event):
    """Traite les requêtes entrantes selon le format spécifié"""
    try:
        # Récupérer les données d'entrée
        input_data = event.get("input", {})
        
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
        
        # Vérifier que le fichier d'entrée existe
        if not os.path.exists(input_path):
            error_msg = f"Le fichier d'entrée {input_path} n'existe pas"
            send_discord_error(
                "Erreur de fichier manquant", 
                error_msg
            )
            return {"error": error_msg}
        
        # Construire et exécuter la commande d'inférence exactement comme spécifié
        cmd = (
            f"python3 /MVSEP-MDX23-music-separation-model/cr.py {output_filename} "
            f"/MVSEP-MDX23-music-separation-model/inference_demucs.py "
            f"--input_audio {input_path} --id_customer {customer_id}"
        )
        
        print(f"Exécution de la commande: {cmd}")
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Vérifier si le processus s'est bien terminé
        if process.returncode != 0:
            error_msg = f"Erreur pendant l'inférence avec code {process.returncode}"
            send_discord_error(
                "Erreur d'inférence", 
                f"Commande: {cmd}\n\nErreur: {process.stderr}",
                process.stderr
            )
            return {
                "error": error_msg,
                "command": cmd,
                "stderr": process.stderr
            }
        
        return {
            "output": {
                "success": True,
                "message": "Traitement audio terminé",
                "output_filename": output_filename,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        }
            
    except Exception as e:
        error_msg = f"Erreur inattendue pendant le traitement: {e}"
        send_discord_error(
            "Erreur inattendue Serverless", 
            error_msg,
            traceback.format_exc()
        )
        return {"error": error_msg}

# Démarrer le service serverless
runpod.serverless.start({
    "handler": handler,
    "init": init
})
