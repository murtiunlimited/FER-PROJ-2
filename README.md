#  Facial Emotion Recognition (FER Project)

This project performs **real-time facial emotion detection** using a CNN model.  
It supports both:
-  OpenCV (desktop webcam)
-  Web browser (FastAPI + frontend)
-  Perform All Instructions From Root Directory

---

## 📁 Project Setup and Structure

```text
FER_PROJECT/
├── .vscode/
├── backend/
│   ├── __pycache__/
│   ├── app.py
│   └── model.py
├── data/
│   ├── processed/
│   │   ├── train/
│   │   └── validation/
│   └── raw/
│       ├── test/
│       └── train/
├── frontend/
│   └── index.html
├── R&D/
│   ├── 1_exp_emotion_cnn_light_v1.py
│   └── 2_exp_emotion_customcnn.py
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── webcam.py
├── venv/
├── best_emotion_model.keras
├── Dockerfile
├── final_emotion_model.keras
├── README.md
└── requirements.txt
```

---

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

#  OpenCV Version (Desktop Webcam)

### Step 1: Preprocess Data
```bash
python -m src.preprocess
```

### Step 2: Train Model
```bash
python -m src.train
```

### Step 3: Run Webcam Detection
```bash
python -m src.webcam
```

---

#  Web Browser Version (FastAPI + Frontend)

### Step 1: Preprocess Data
```bash
python -m src.preprocess
```

### Step 2: Train Model
```bash
python -m src.train
```

### Step 3: Start Backend Server
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Launch Frontend
- Open `frontend/index.html` in your browser  
- Allow camera access  
- Start detecting emotions 🎉

---

##  Model Details
- Input: 48×48 grayscale face images  
- Architecture: Lightweight CNN  
- Classes:
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  

---

##  Features
- Real-time emotion detection  
- Lightweight CNN (fast inference)  
- Works with webcam + browser  
- FastAPI backend for scalable deployment  

---

##  Notes
- Ensure your webcam is accessible  
- Backend must be running before opening the frontend  
- Model file (`final_emotion_model.keras`) must exist in root directory  
