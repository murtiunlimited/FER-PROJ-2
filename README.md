#  Facial Emotion Recognition (FER Project)

This project performs **real-time facial emotion detection** using a CNN model.  
It supports both:
-  OpenCV (desktop webcam)
-  Web browser (FastAPI + frontend)

---

## 📁 Project Setup and Structure

```text
FER-PROJ-2/
├── api/
│   ├── __pycache__/
│   └── app.py
├── data/
│   ├── processed/
│   │   ├── train/
│   │   └── validation/
│   └── raw/
│       ├── test/
│       └── train/
├── frontend/
│   └── index.html
├── models/
│   ├── best_emotion_model.keras
│   └── final_emotion_model.keras
└── src/
    ├── __pycache__/
    ├── data/
    │   ├── __pycache__/
    │   ├── __init__.py
    │   ├── preprocess.py
    │   └── split.py
    ├── inference/
    │   ├── __pycache__/
    │   ├── __init__.py
    │   ├── predict.py
    │   └── webcam.py
    ├── models/
    │   ├── __pycache__/
    │   ├── __init__.py
    │   ├── evaluate.py
    │   ├── model.py
    │   └── train.py
    ├── utils/
    │   ├── __init__.py
    │   └── config.py
    └── __init__.py
├── venv/
├── Dockerfile
├── README.md
├── requirements.txt
├── run_pipeline.py
└── shellscript.sh
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

### 1. Preprocess Data
```bash
python -m src.preprocess
```

### 2. Train Model
```bash
python -m src.train
```

### 3. Run Webcam Detection
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
uvicorn api.app:app --reload
```

### Step 4: Launch Frontend
- Open `frontend/index.html` in your browser  
- Allow camera access  
- Start detecting emotions 🎉

---

#  Web Browser Version With Shell

### Enable Script Execution
```bash
chmod +x shellscript.sh   
```

### Run Full Pipeline (Preprocess + Train + Launch)
```bash
./shellscript.sh
```

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
