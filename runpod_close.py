import os
import requests
import json
import sys

# Configuration - Récupération depuis les variables d'environnement
API_KEY = os.environ.get("RUNPOD_API_KEY")
POD_ID = os.environ.get("RUNPOD_POD_ID")

def main():
    # Votre programme principal ici
    print("Exécution du programme principal...")
    
    # Exemple : effectuer un traitement
    # ...
    
    print("Programme principal terminé")

def terminate_pod_v1_rest():
    """Tente de terminer le pod via l'API REST v1"""
    print("\nMéthode 1: API REST v1...")
    url = f"https://api.runpod.io/v1/pod/{POD_ID}/terminate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(url, headers=headers)
        print(f"Code de statut: {response.status_code}")
        print(f"Réponse: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return False

def terminate_pod_v2_graphql():
    """Tente de terminer le pod via l'API GraphQL v2"""
    print("\nMéthode 2: API GraphQL...")
    url = "https://api.runpod.io/v2/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "query": """
        mutation($podId: String!) {
            terminatePod(podId: $podId) {
                success
                message
            }
        }
        """,
        "variables": {
            "podId": POD_ID
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Code de statut: {response.status_code}")
        print(f"Réponse: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return False

def terminate_pod_direct_command():
    """Tente de terminer le pod via une commande système"""
    print("\nMéthode 3: Commande système directe...")
    try:
        import subprocess
        cmd = f"curl -s -X DELETE https://api.runpod.io/v2/pod/{POD_ID} -H 'Authorization: Bearer {API_KEY}'"
        print(f"Exécution de: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Sortie: {result.stdout}")
        print(f"Erreur: {result.stderr}")
        return True
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Exécuter votre programme principal
        main()
    finally:
        print(f"\nTentative de suppression du pod {POD_ID}...")
        
        # Vérifier si les variables d'environnement sont définies
        if not POD_ID:
            print("ERREUR: Variable d'environnement RUNPOD_POD_ID non définie")
            sys.exit(1)
        
        if not API_KEY:
            print("ERREUR: Variable d'environnement RUNPOD_API_KEY non définie")
            sys.exit(1)
        
        # Essayer toutes les méthodes
        if terminate_pod_v1_rest():
            print("✅ Pod supprimé avec succès (méthode 1)")
        elif terminate_pod_v2_graphql():
            print("✅ Pod supprimé avec succès (méthode 2)")
        elif terminate_pod_direct_command():
            print("✅ Pod supprimé avec succès (méthode 3)")
        else:
            print("❌ Toutes les méthodes ont échoué")
