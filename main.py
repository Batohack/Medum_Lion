from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI(title="Deepfake Detector API", version="1.0")

origins = [
    "*", # DANGER : En prod, on mettrait l'URL précise du site (ex: https://monsite.com)
]    
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Définition de l'architecture du modèle (doit être identique à train_model.py) ---
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

# --- 2. Chargement du modèle au démarrage du serveur ---
model = SimpleCNN()
try:
    model.load_state_dict(torch.load("best_medumlion_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")

# --- 3. Configuration de la transformation d'image ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# --- 4. La Route API (Le point d'entrée) ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lecture du fichier uploadé
    image_bytes = await file.read()
    
    # Préparation
    tensor = transform_image(image_bytes)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 0 = fake, 1 = real (selon l'ordre alphabétique de vos dossiers)
        score_fake = probabilities[0][0].item()
        score_real = probabilities[0][1].item()
        
        pred_class = "REAL" if score_real > score_fake else "FAKE"
        confidence = max(score_real, score_fake)

    return {
        "filename": file.filename,
        "prediction": pred_class,
        "confidence": f"{confidence:.2%}",
        "scores": {
            "fake_probability": score_fake,
            "real_probability": score_real
        }
    }

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de détection de Deepfake. Allez sur /docs pour tester."}