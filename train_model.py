import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DIR = "/home/batohack/Téléchargements/dataset"
BATCH_SIZE = 32     # On garde des petits paquets pour le CPU
IMG_SIZE = 128
EPOCHS = 25         # ON AUGMENTE ! (Prévoyez une nuit sur CPU)
MODEL_SAVE_PATH = "best_medumlion_model.pth" # Nom du fichier du meilleur modèle

# Détection automatique du matériel (sera 'cpu' dans votre cas)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement sur le matériel : {device}")

# ==========================================
# ÉTAPE 1 : Pipeline de données Amélioré (Data Augmentation)
# ==========================================
print("Préparation des données avec Augmentation...")

# C'est ici que la magie opère : on rajoute de l'aléatoire
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # --- NOUVEAUTÉS PRO ---
    transforms.RandomHorizontalFlip(p=0.5), # 50% de chance de retourner l'image
    transforms.RandomRotation(degrees=15),  # Rotation légère aléatoire (+/- 15°)
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Légère variation de lumière
    # ----------------------
    transforms.ToTensor(),
])

try:
    # Chargement du dataset avec les nouvelles transformations
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Classes : {full_dataset.classes}")
    print(f"Images totales : {len(full_dataset)}")
    print("DataLoaders prêts.")

except Exception as e:
    print(f"Erreur critique au chargement des données : {e}")
    exit()

# ==========================================
# ÉTAPE 2 : Le Cerveau (Inchangé, l'architecture est bonne)
# ==========================================
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

# ==========================================
# ÉTAPE 3 : Initialisation
# ==========================================
model = SimpleCNN().to(device) # On envoie le modèle sur le CPU
criterion = nn.CrossEntropyLoss()
# On réduit légèrement le taux d'apprentissage (learning rate) pour être plus précis
optimizer = optim.Adam(model.parameters(), lr=0.0005) 

# Variable pour retenir le meilleur score jamais atteint
best_loss = float('inf') 

# ==========================================
# ÉTAPE 4 : La Boucle d'Entraînement PRO
# ==========================================
print(f"\n🔥 DÉBUT DE L'ENTRAÎNEMENT LONG ({EPOCHS} Epochs) 🔥")
print("C'est le moment d'aller dormir... ça va être long sur CPU.\n")

for epoch in range(EPOCHS):
    model.train() # Mode entraînement activé
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # On déplace les données sur le CPU
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Affichage de progression (tous les 20 batchs pour moins spammer)
        if i % 20 == 19:
            print(f'[Epoch {epoch + 1}/{EPOCHS}, Batch {i + 1}] loss en cours: {running_loss / 20:.4f}')
            running_loss = 0.0
            
    # --- FIN DE L'EPOCH : Calcul du bilan et Checkpoint ---
    # On calcule la perte moyenne sur toute l'époque pour voir si on s'améliore
    # (Calcul simplifié ici sur le training set pour ce tutoriel)
    epoch_loss = loss.item() 

    print(f"--> Fin Epoch {epoch + 1}. Loss finale: {epoch_loss:.4f}")

    # CHECKPOINT : Est-ce que c'est le meilleur modèle jusqu'à présent ?
    if epoch_loss < best_loss:
        print(f"✅ AMÉLIORATION DÉTECTÉE ! (Ancien best: {best_loss:.4f} -> Nouveau: {epoch_loss:.4f})")
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Modèle sauvegardé sous : {MODEL_SAVE_PATH}")
    else:
        print(f"Pas d'amélioration ce tour-ci (Best reste: {best_loss:.4f})")
    
    print("-" * 50)

print("\n✅ ENTRAÎNEMENT TERMINÉ !")
print(f"Le meilleur modèle a été sauvegardé dans : {MODEL_SAVE_PATH}")
print("Vous pouvez maintenant utiliser ce fichier avec votre API.")