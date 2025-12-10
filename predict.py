import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# 1. On définit la même architecture que lors de l'entraînement
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Fonction pour charger le modèle
def load_model(model_path):
    model = SimpleCNN()
    # On charge les poids entraînés
    model.load_state_dict(torch.load(model_path))
    model.eval() # Important : met le modèle en mode "Examen" (pas "Entraînement")
    return model

# 3. Fonction pour préparer l'image (comme dans train_model.py)
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        # On ajoute une dimension "Batch" car le modèle attend [1, 3, 128, 128]
        return transform(image).unsqueeze(0) 
    except Exception as e:
        print(f"Erreur avec l'image : {e}")
        return None

# ================= MAIN =================
if __name__ == "__main__":
    # Vérifie si l'utilisateur a donné un fichier image
    if len(sys.argv) < 2:
        print("Usage : python predict.py <chemin_vers_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "deepfake_model.pth"

    # Chargement
    print("Chargement du cerveau...")
    model = load_model(model_path)

    # Préparation
    img_tensor = process_image(image_path)
    
    if img_tensor is not None:
        # Prédiction
        with torch.no_grad(): # Pas besoin de calculer les gradients pour une prédiction
            outputs = model(img_tensor)
            # On applique Softmax pour avoir des pourcentages
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Récupérer la classe gagnante
            _, predicted = torch.max(outputs, 1)
            
            # IMPORTANT : L'ordre dépend de l'ordre alphabétique des dossiers
            # 0 = training_fake, 1 = training_real
            classes = ["FAKE (Généré par IA)", "REAL (Authentique)"]
            
            score_fake = probabilities[0][0].item() * 100
            score_real = probabilities[0][1].item() * 100

            print(f"\n--- RÉSULTAT ---")
            print(f"Image analysée : {image_path}")
            print(f"Verdict : {classes[predicted.item()]}")
            print(f"Confiance : Fake={score_fake:.2f}% | Real={score_real:.2f}%")