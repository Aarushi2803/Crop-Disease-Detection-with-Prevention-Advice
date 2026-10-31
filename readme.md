# 🌿 CropDoc: Crop Disease Detection with Prevention Advice

Upload (or capture) a photo of a plant leaf. CropDoc identifies the disease with a deep learning model, then tells the farmer **what it is, why it happens, how to treat it, and how to prevent it from coming back**.

**Live demo:** _TBD (add your Hugging Face Spaces link here)_

> **Screening aid only.** Advice is general guidance. Confirm with your local Krishi Vigyan Kendra (KVK) or agriculture officer before applying any chemical, and always follow the product label.

<!-- Add 2-3 screenshots here: home screen, a result card, the "not sure" screen -->

---

## Why this project

- Crop diseases cause large yield losses for Indian farmers, and expert help is often far away.
- Models that score highly on the clean PlantVillage dataset often do worse on real field photos, because lab images have plain backgrounds. This project measures that gap on real photos instead of hiding it.
- Detection alone is not enough. The farmer needs a next step, so every prediction is linked to a prevention and treatment knowledge base.

## Features

- Leaf photo to disease prediction across **38 classes**, with the **top 3 guesses and confidence**
- **Low-confidence safeguard:** below 60% confidence the app says "not sure, retake the photo" and withholds advice instead of guessing
- Symptoms, causes, treatment and prevention advice for every class, plus a safety disclaimer
- Mobile-first website with direct camera capture and a small "recent scans" list (stored only in the user's browser)
- Lightweight **ONNX** model (MobileNetV3-Large) that runs on CPU, so no GPU is needed to serve it
- REST API (`/api/predict`) so other apps can use it later
- Uploaded photos are processed in memory and **not stored** on the server
- Automated tests and a Docker image for reproducible deployment

## How it works

```
Phone/browser ──photo──▶ FastAPI (/api/predict)
                           │ 1. validate image (type, size ≤ 8 MB)
                           │ 2. fix rotation, resize 256, centre-crop 224, normalise
                           │ 3. ONNX Runtime inference (MobileNetV3-Large)
                           │ 4. softmax → top-3, low-confidence flag (< 0.60)
                           │ 5. look up advice in data/prevention.json
                           ▼
              JSON: top guess, top 3, advice, disclaimer ──▶ result card in the browser
```

Training and serving use the **same preprocessing** (resize 256, centre crop 224, ImageNet mean/std). A mismatch here silently lowers accuracy, so it is kept identical in `src/train.py` and `app/main.py`.

## Project status

| Phase | Goal | Status |
|---|---|---|
| 1. Setup & data | PlantVillage in ImageFolder layout, class balance check, train/valid split | ✅ Done |
| 2. Baseline model | Fine-tune pretrained MobileNetV3-Large | ✅ Pipeline works, final run: _TBD_ |
| 3. Real-world gap test | Evaluate on real phone photos | ⏳ _TBD_ |
| 4. Robustness | Heavy augmentation, label smoothing (implemented); comparison against baseline | ⏳ _TBD_ |
| 5. Evaluation | Per-class precision/recall/F1 (`runs/report.txt`); Grad-CAM planned | ⏳ In progress |
| 6. Knowledge base | 38 classes of cause / symptoms / treatment / prevention | ✅ Drafted, expert review pending |
| 7. Backend API | FastAPI + ONNX Runtime | ✅ Done |
| 8. Website | Responsive UI with camera upload | ✅ Done |
| 9. Deployment | Docker, Hugging Face Spaces | ⏳ _TBD_ |

## Project structure

```
CropDoc/
├── README.md
├── requirements.txt          # full environment for training
├── requirements-deploy.txt   # minimal server dependencies (used by Docker)
├── requirements-dev.txt      # server + test tools
├── Dockerfile
├── src/
│   ├── train.py              # training + evaluation + ONNX export
│   ├── export_checkpoint.py  # export runs/best.pt to ONNX without retraining
│   └── split_valid.py        # create a fixed-seed train/valid split
├── data/
│   └── prevention.json       # class → cause, symptoms, prevention, treatment
├── models/                   # model.onnx, classes.json (created by train.py)
├── tests/
│   ├── test_api.py           # API tests in demo mode
│   └── test_data.py          # checks the knowledge base
└── app/
    ├── main.py               # FastAPI backend
    └── static/               # index.html, style.css, script.js
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate           # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The Kaggle **New Plant Diseases Dataset (Augmented)**, a version of PlantVillage with 38 classes, in an ImageFolder layout (one folder per class). Place it in the project root as `train/` and `valid/` (both are git-ignored because of their size):

```
train/Apple___Apple_scab/...      valid/Apple___Apple_scab/...
```

Class folder names must match the keys in `data/prevention.json`.

If you only have a `train/` folder, `python src/split_valid.py` moves a fixed-seed 20% of each class into `valid/`. Note: the augmented dataset contains rotated and flipped copies of the same photos, so a random split can place near-duplicates on both sides and make validation accuracy look better than it is. For this reason the **real-photo test below is the number to trust**.

## Train

```powershell
# CPU-friendly run on a subset (300 images per class)
python src/train.py --data . --epochs 5 --subset 300 --batch-size 32 --light-aug --workers 2

# Full run (use a GPU, e.g. free Kaggle or Colab)
python src/train.py --data . --epochs 8 --batch-size 64 --workers 4
```

| Flag | Meaning |
|---|---|
| `--subset N` | use only N images per class (quick tests) |
| `--light-aug` | light augmentation (baseline). The default is heavy augmentation for robustness to field photos |
| `--workers N` | data loading processes (use `0` if Windows gives errors) |
| `--epochs`, `--batch-size`, `--lr`, `--patience` | usual training settings, with early stopping |

Training writes `models/model.onnx`, `models/classes.json`, `runs/best.pt` and `runs/report.txt`, then checks that the ONNX output matches PyTorch. To export the best saved epoch after stopping early: `python src/export_checkpoint.py`.

## Run the website

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. To try the UI without a trained model:

```powershell
$env:DEMO_MODE=1; uvicorn app.main:app --reload      # Mac/Linux: DEMO_MODE=1 uvicorn app.main:app --reload
```

Demo mode returns **fake, clearly labelled** predictions. Never use it for real advice.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service and model status |
| `/api/predict` | POST (multipart `file`) | Top-3 predictions and advice for the top result |
| `/api/diseases` | GET | All supported classes |

Example response:

```json
{
  "demo_mode": false,
  "top": {"class": "Tomato___Late_blight", "label": "Tomato: Late blight", "confidence": 0.93},
  "predictions": [ "... 3 items ..." ],
  "low_confidence": false,
  "advice": {"crop": "...", "disease": "...", "healthy": false, "cause": "...", "symptoms": "...", "prevention": ["..."], "treatment": ["..."]},
  "disclaimer": "General guidance only. Confirm with your local KVK ..."
}
```

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -v
```

Covers the API (health, validation, size limit, top-3 shape, static file types) and the knowledge base (all 38 classes complete, and every model class has advice).

## Results

| Experiment | Validation acc (PlantVillage) | Real phone photos | Notes |
|---|---|---|---|
| Baseline (light augmentation) | _TBD_ | _TBD_ | subset / full: _TBD_ |
| + Heavy augmentation | _TBD_ | _TBD_ | |

**How the real-photo number is measured:** _TBD_ photos (single leaf, daylight, mixed healthy and diseased) uploaded to the app; accuracy = correct top-1 ÷ total. Add a short note on which classes were confused (see `runs/report.txt`).

## Docker and deployment

```powershell
docker build -t cropdoc .
docker run -p 7860:7860 cropdoc
```

Open http://127.0.0.1:7860. The image installs only `requirements-deploy.txt` (no PyTorch), runs as a non-root user, and reads the port from the environment, so it works on **Hugging Face Spaces** (Docker Space, port 7860), Render, or any VPS. `models/model.onnx` and `models/classes.json` must be included in the image.

## Limitations

- Trained on PlantVillage-style lab images covering 38 classes (apple, tomato, potato, corn, grape, etc.). Important Indian crops such as rice, wheat, cotton and sugarcane are **not** covered.
- Accuracy on real field photos is lower than on lab images, which is why low-confidence results are flagged and advice is withheld.
- Validation accuracy can be optimistic because of near-duplicate augmented images (see Dataset).
- One leaf per image and one disease per leaf; multiple problems at once are not handled.
- The advice text is general and still needs expert review.
- **A screening aid, not a replacement for an agricultural expert.**

## Roadmap

Grad-CAM heatmaps, real field-photo datasets such as PlantDoc mixed into training, Hindi and Kannada advice, offline use (PWA or TFLite), and extra Indian crops.

## Credits

- PlantVillage dataset (Hughes & Salathé, 2015), via the Kaggle "New Plant Diseases Dataset (Augmented)"
- Pretrained weights: torchvision MobileNetV3-Large (ImageNet)