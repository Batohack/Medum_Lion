# Medum Lion — Deepfake Detector

Une application full‑stack pour détecter les images manipulées par IA (deepfakes).  
Combinaison d’un backend FastAPI + PyTorch et d’une interface web légère pour analyser des images en temps réel.

---

## ✅ Résumé rapide
- Nom : **Medum Lion**
- Fonction : Détecter si une image est REAL (humaine) ou FAKE (générée/manipulée par IA)
- Tech stack : `Python 3.11`, `PyTorch`, `FastAPI`, `Uvicorn`, frontend HTML/CSS/JS
- Usage principal : démonstration locale / prototype de détection d’image

---

## 🚀 Quickstart (en local)
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Batohack/Medum_Lion.git
   cd Medum_Lion
   ```
2. Créer et activer l'environnement virtuel :
   ```bash
   python3.11 -m venv medumlion
   source medumlion/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
4. Lancer l'API :
   ```bash
   uvicorn main:app --reload
   ```
5. Ouvrir l'interface :  
   Navigateur → `http://127.0.0.1:8000` (ou ouvrez `index.html` pour test local)

---

## 📁 Structure du projet (essentiel)
- `main.py` — serveur FastAPI + endpoint `/predict/` pour l’inférence  
- `train_model.py` — script d’entraînement (SimpleCNN)  
- `best_medumlion_model.pth` — poids du modèle (pré-entrainé)  
- `index.html` — frontend (UI + Canvas Matrix animation)  
- `requirements.txt` — dépendances  
- `check_data.py` — utilitaire de vérification des datasets

---

## 🧠 Modèle & pipeline
- Modèle : `SimpleCNN` (3 couches conv → flatten → FC → 2 sorties)
- Prétraitement : resize 128×128 → ToTensor
- Sortie : Softmax → probabilités `{ fake_probability, real_probability }`
- Résultat renvoyé : `{ filename, prediction: "REAL"|"FAKE", confidence: "xx.xx%" }`

---

## 🎨 Design & animation
L’interface intègre un arrière‑plan animé "Matrix" (HTML5 Canvas) :
- Caractères verts tombant du haut vers le bas (0/1 + kanji)
- Trail semi‑transparent pour l’effet visuel
- Conteneur principal en verre dépoli (`backdrop-filter: blur(10px)`)
- Animation en JS vanilla (`requestAnimationFrame`) — sans dépendances externes

---

## 🛠️ Dépannage rapide
- Erreur d’espace disque lors de l’installation (Errno 122) :
  ```bash
  TMPDIR=/tmp pip install --no-cache-dir -r requirements.txt
  ```
  ou libérer de l’espace dans `~/.cache` / `~/Téléchargements`.
- Port 8000 occupé → tuer le process ou utiliser un autre port :
  ```bash
  lsof -i :8000
  kill -9 <PID>
  ```
- Modèle introuvable → réentraîner :
  ```bash
  python train_model.py
  ```

---

## 🔐 Conseils pour la production
- Restreindre CORS (ne pas utiliser `allow_origins = ["*"]` en prod)
- Ajouter authentification (JWT/OAuth) sur `/predict/`
- Limiter la taille des images uploadées pour éviter DoS
- Conteneuriser avec Docker et surveiller les ressources

---

## 📚 Ressources & évolutions prévues
- Support vidéo (frame‑by‑frame)
- Fine‑tuning avec backbone pré‑entrainés (ResNet, EfficientNet)
- Dashboard statistiques & historique des prédictions
- Déploiement Docker + CI (GitHub Actions)

---

## ♻️ Contribuer
1. Fork → créer une branche feature → PR
2. Respecter la structure du projet et ajouter tests si possible
3. Ouvrir une issue / discussion pour les gros changements

---

## Licence & contact
- Licence : **MIT**  
- Auteur / contact : **Batohack** — voir le repo GitHub

---

*Page générée automatiquement depuis le dépôt local.*
