import torch
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