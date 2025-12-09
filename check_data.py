import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = "/home/batohack/Téléchargements/archive"

CATEGORIES = ["training_fake", "training_real"]

print(f"Recherche des données dans : {os.path.abspath(BASE_DIR)}")
for category in CATEGORIES:
    path = os.path.join(BASE_DIR, category)

    if not os.path.exists(path):
        print(f"Erreur : Le dossier  {path} n'existe pas. Veuillez vérifier le chemin dans CATEGORIES.")
        continue
    print("--- Analyse du dossier : {category} ---")

    images_list = os.listdir(path)
    if not images_list:
        print("Dossier vide !")
        continue

    first_image_name = images_list[0]
    full_path = os.path.join(path, first_image_name)

    try:
        img_array = cv2.imread(full_path)
        if img_array is None:
            print("Impossible de lire l'imaage avec OpenCV.")
            continue
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.title(f"Exemple : {category}")
        plt.show()
    
        break  # Sortir de la boucle après avoir affiché une image
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image : {e}")