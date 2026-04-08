# src/data/preprocess.py

import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
RAW_DIR = "data/raw/train"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (48, 48)
VAL_SPLIT = 0.1  # 10% of training for validation

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# -------------------------
# Helpers
# -------------------------
def create_dirs():
    """Create processed/train & processed/validation directories."""
    for split in ["train", "validation"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

def preprocess_and_save():
    create_dirs()
    for cls in CLASS_NAMES:
        cls_path = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        train_imgs, val_imgs = train_test_split(images, test_size=VAL_SPLIT, random_state=42)

        for split, img_list in zip(["train", "validation"], [train_imgs, val_imgs]):
            for img_name in tqdm(img_list, desc=f"Processing {cls} - {split}"):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # normalize
                save_path = os.path.join(PROCESSED_DIR, split, cls, img_name)
                cv2.imwrite(save_path, (img * 255).astype('uint8'))

if __name__ == "__main__":
    preprocess_and_save()
    print("✅ Preprocessing complete! Images saved to data/processed/")