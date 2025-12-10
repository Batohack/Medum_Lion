import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DIR = "/home/batohack/Téléchargements/dataset"
BATCH_SIZE = 32
IMG_SIZE = 128

data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # On force la taille
    transforms.ToTensor(),                   # On convertit en Tenseur PyTorch (0-1)
])

try:
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform = data_transform)

    print(f"Classes trouvées : {full_dataset.classes}")
    print(f"Total d'images : {len(full_dataset)}")

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    images, labels = next(iter(train_loader))

    print(f"Taille du batch d'images : {images.shape}")


    print("Le Dashloader fonctionne correctement.")

except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    print("Veuillez vérifier le chemin DATA_DIR et la structure des dossiers.")


# ==========================================
# ÉTAPE 5 : Construction du Cerveau (CNN)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Couche 1 : Analyse les formes simples (lignes, courbes)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Divise la taille de l'image par 2
        
        # Couche 2 : Analyse des formes plus complexes (yeux, nez)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Couche 3 : Analyse globale
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Couche finale : Classification (Vrai ou Faux)
        # 128 canaux * 16 * 16 (taille finale de l'image après 3 pools)
        self.fc1 = nn.Linear(128 * 16 * 16, 512) 
        self.fc2 = nn.Linear(512, 2) # 2 sorties : REAL ou FAKE

    def forward(self, x):
        # Passage dans les couches de convolution
        x = self.pool(self.relu(self.conv1(x))) # Image devient 64x64
        x = self.pool(self.relu(self.conv2(x))) # Image devient 32x32
        x = self.pool(self.relu(self.conv3(x))) # Image devient 16x16
        
        # Aplatissement (On transforme le cube 3D en ligne plate)
        x = x.view(-1, 128 * 16 * 16)
        
        # Décision finale
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# ÉTAPE 6 : Lancer l'entraînement
# ==========================================

# 1. Création de l'IA
print("Création du modèle...")
model = SimpleCNN()

# 2. Définition des outils d'apprentissage
criterion = nn.CrossEntropyLoss() # Pour calculer l'erreur
optimizer = optim.Adam(model.parameters(), lr=0.001) # Pour corriger l'erreur

# 3. Boucle d'entraînement (Loop)
EPOCHS = 3 # On va lire tout le dataset 3 fois (Commencez petit car vous êtes sur CPU)

print("Début de l'entraînement... (Patientez, ça chauffe le CPU !)")

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        # Remise à zéro des gradients
        optimizer.zero_grad()
        
        # A. L'IA fait une prédiction
        outputs = model(inputs)
        
        # B. On calcule l'erreur (Loss)
        loss = criterion(outputs, labels)
        
        # C. On corrige l'IA (Backpropagation)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Affichage tous les 10 batchs pour voir que ça avance
        if i % 10 == 9: 
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Terminé ! Modèle entraîné.')

# 4. Sauvegarde du cerveau
torch.save(model.state_dict(), 'deepfake_model.pth')
print("Modèle sauvegardé sous 'deepfake_model.pth'")