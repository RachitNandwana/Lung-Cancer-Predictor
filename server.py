# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# CONFIG
HF_REPO_ID = "Rachit02/lung-cancer-prediction"   # your HF repo
HF_FILENAME = "lung_cancer_model_finalmodel.h5"  # exact filename in repo
HF_TOKEN = os.environ.get("HF_TOKEN")            # set as env var on host
MODEL_LOCAL_PATH = "model.h5"

app = FastAPI(title="Lung Cancer Inference API")

def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        return MODEL_LOCAL_PATH
    try:
        local_file = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, use_auth_token=HF_TOKEN)
        # copy or point to the downloaded local_file
        # hf_hub_download already returns a cache path we can use directly
        return local_file
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HF: {e}")

# load model once
MODEL_PATH = download_model_if_needed()
print("Loading Keras model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

def preprocess_image_bytes(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        X = preprocess_image_bytes(content, target_size=(224,224))  # adjust size to your model
        preds = model.predict(X)  # shape depends on your model
        # interpret predictions (adjust logic to your model's output)
        if preds.ndim == 2 and preds.shape[1] == 2:
            prob = float(preds[0][1])
            label = int(np.argmax(preds[0]))
        else:
            prob = float(preds[0][0])
            label = 1 if prob >= 0.5 else 0
        return JSONResponse({"prediction": int(label), "probability": float(prob)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status":"ok"}
