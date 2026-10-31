"""Turn the best saved checkpoint (runs/best.pt) into models/model.onnx WITHOUT retraining.

Use it when you stop training early (Ctrl+C) after at least one epoch has finished,
or if the export step at the end of train.py ever fails.

Run from the PROJECT ROOT:
    python src/export_checkpoint.py
"""
import json
from pathlib import Path

import torch

# Reuse the exact same functions as train.py so the model and the export match
from train import build_model, export_onnx, get_loaders, verify_onnx


def main():
    ckpt = Path("runs/best.pt")
    if not ckpt.exists():
        raise SystemExit("runs/best.pt not found. Let training finish at least one full epoch first.")

    # A tiny loader (a few images per class) just to read the class names and check the export
    classes, _, val_dl = get_loaders(Path("."), batch_size=16, workers=0, light_aug=True, subset=4)

    model = build_model(len(classes), pretrained=False)          # same layers, no download needed
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))  # load the saved weights
    model.eval()

    out = Path("models")
    out.mkdir(exist_ok=True)
    (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    export_onnx(model, out / "model.onnx")
    verify_onnx(model, out / "model.onnx", val_dl)
    print("Saved models/model.onnx and models/classes.json. Restart uvicorn to use them.")


if __name__ == "__main__":
    main()
    