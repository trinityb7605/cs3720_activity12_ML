from io import BytesIO
from time import perf_counter

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

# Limit CPU threads for this initial deployment test.
torch.set_num_threads(1)

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

app = FastAPI(title="MobileNetV3 Small Image Classifier")

@app.get("/")
def home():
    return {
        "status": "ready",
        "model": "MobileNetV3 Small",
        "categories": len(categories),
    }

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Upload a valid JPEG or PNG image.",
        )

    start = perf_counter()
    batch = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]
        scores, indices = probabilities.topk(3)

    predictions = [
        {
            "label": categories[index],
            "score": round(score * 100, 2),
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]

    return {
        "predictions": predictions,
        "inference_ms": round((perf_counter() - start) * 1000, 1),
    }
