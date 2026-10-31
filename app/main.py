"""CropDoc backend: FastAPI + ONNX Runtime.              
Run normally : uvicorn app.main:app --reload

"""
import io
import json
import mimetypes
import os
import random
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError 

# app/main.py

app = FastAPI()

@app.get("/")
def read_index():
    return FileResponse(os.path.join("app", "static", "index.html"))

# ---------- File locations ----------
ROOT = Path(__file__).resolve().parent.parent            # project folder
STATIC = Path(__file__).resolve().parent / "static"      # app/static
MODEL_PATH = ROOT / "models" / "model.onnx"              # made by train.py
CLASSES_PATH = ROOT / "models" / "classes.json"          # made by train.py
PREVENTION_PATH = ROOT / "data" / "prevention.json"      # advice for each disease

# ---------- Settings ----------
DEMO_MODE = os.getenv("DEMO_MODE") == "1"    # fake predictions if True
LOW_CONFIDENCE = 0.60                        # below this, tell the user to retake the photo
MAX_UPLOAD_BYTES = 8 * 1024 * 1024           # reject images larger than 8 MB
IMG_SIZE = 224                               # must match train.py
RESIZE_TO = 256                              # must match the validation transform in train.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet mean (must match train.py)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # ImageNet std  (must match train.py)

# Windows can map .css to the wrong type (text/plain), and then the browser ignores the stylesheet.
# Setting the types ourselves makes the site look the same on Windows, Linux and Docker.
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")

app = FastAPI(title="CropDoc", version="1.0.0")

# ---------- Load advice once at startup ----------
prevention = json.loads(PREVENTION_PATH.read_text(encoding="utf-8"))
DISCLAIMER = prevention.pop("_disclaimer", "")   # keep the safety note, it is not a class

# ---------- Load the model once at startup ----------
session = None       # the ONNX model (None until loaded)
INPUT_NAME = None    # the model's input name, read from the file itself
classes = []         # class names in the same order the model was trained with

if DEMO_MODE:
    classes = sorted(prevention.keys())          # no model needed in demo mode
elif MODEL_PATH.exists() and CLASSES_PATH.exists():
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    INPUT_NAME = session.get_inputs()[0].name    # never hardcode this, read it
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    # Stop early if any class has no advice, so we never show wrong or empty advice
    missing = [c for c in classes if c not in prevention]
    if missing:
        raise RuntimeError(f"No advice in prevention.json for: {missing}")


def preprocess(img: Image.Image) -> np.ndarray:
    """Turn a PIL image into the exact tensor the model was trained on."""
    img = ImageOps.exif_transpose(img).convert("RGB")      # fix phone rotation, force 3 colour channels
    w, h = img.size
    scale = RESIZE_TO / min(w, h)                          # make the short side 256
    img = img.resize((max(IMG_SIZE, round(w * scale)), max(IMG_SIZE, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2   # centre crop 224x224
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD   # scale to 0-1, then normalise
    return arr.transpose(2, 0, 1)[None].astype(np.float32)           # HWC -> CHW, add batch dimension


def softmax(x: np.ndarray) -> np.ndarray:
    """Turn raw model scores into probabilities that add up to 1."""
    e = np.exp(x - x.max())
    return e / e.sum()


def pretty(cls: str) -> str:
    """Friendly name like 'Tomato: Late blight' for a class key."""
    info = prevention.get(cls)
    if info:
        return f"{info['crop']}: {info['disease']}"
    return cls.replace("___", ": ").replace("_", " ")


def run_model(img: Image.Image):
    """Run the ONNX model and return the 3 most likely (class, probability) pairs."""
    logits = session.run(None, {INPUT_NAME: preprocess(img)})[0][0]
    if len(logits) != len(classes):
        raise RuntimeError("Model output size does not match classes.json")
    probs = softmax(logits)
    top_idx = np.argsort(probs)[::-1][:3]
    return [(classes[i], float(probs[i])) for i in top_idx]


# ---------- API routes ----------
@app.get("/api/health")
def health():
    """Quick check that the server and model are alive."""
    return {"status": "ok", "model_loaded": session is not None,
            "demo_mode": DEMO_MODE, "classes": len(classes)}


@app.get("/api/diseases")
def diseases():
    """List every disease the app can recognise."""
    return [{"class": c, "label": pretty(c)} for c in classes]


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Receive a leaf photo, return top-3 guesses plus advice."""
    if session is None and not DEMO_MODE:
        raise HTTPException(503, "Model not found. Train it first (see README) or run with DEMO_MODE=1.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB).")

    try:                                         # make sure it really is an image
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "That file is not a valid image.")

    if DEMO_MODE:                                # fake result, clearly marked in the response
        picks = random.sample(classes, 3)
        raw = sorted((random.random() for _ in picks), reverse=True)
        top3 = [(c, r / sum(raw)) for c, r in zip(picks, raw)]
    else:
        top3 = await run_in_threadpool(run_model, img)   # keep the server responsive while predicting

    best_cls, best_p = top3[0]
    return {
        "demo_mode": DEMO_MODE,
        "top": {"class": best_cls, "label": pretty(best_cls), "confidence": round(best_p, 4)},
        "predictions": [{"class": c, "label": pretty(c), "confidence": round(p, 4)} for c, p in top3],
        "low_confidence": best_p < LOW_CONFIDENCE,   # True means "ask the user to retake the photo"
        "advice": prevention.get(best_cls),
        "disclaimer": DISCLAIMER,
    }


# ---------- Website ----------
@app.get("/")
def index():
    """Serve the main page."""
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")   # css and js files
