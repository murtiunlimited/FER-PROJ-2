# api/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# Import the interface prediction function
from src.inference.predict import predict_emotion

# =========================
# FastAPI app
# =========================
app = FastAPI(title="FER Emotion Detection API")

# Allow requests from any origin (for local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # grayscale

    # Predict emotion using interface function
    label = predict_emotion(img)

    return {"emotion": label}