"""Train a MobileNetV3 crop-disease classifier and export it to ONNX.

Run from the PROJECT ROOT (the folder that contains train/, valid/, src/, app/):

    python src/train.py --data . --epochs 1 --subset 20 --batch-size 32     # quick smoke test
    python src/train.py --data . --epochs 12 --batch-size 64                 # real run (use a GPU)
    python src/train.py --data . --epochs 12 --light-aug                     # baseline (light augmentation)

Expected layout: <data>/train/<class>/*.jpg and <data>/valid/<class>/*.jpg
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# These must match app/main.py (resize 256, centre crop 224, ImageNet mean/std)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def per_class_subset(dataset, n_per_class, seed=42):
    """Keep only n images per class so a smoke test runs in minutes."""
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(dataset.targets):
        by_class.setdefault(label, []).append(idx)
    keep = []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        keep.extend(idxs[:n_per_class])
    return Subset(dataset, sorted(keep))


def get_loaders(data_dir: Path, batch_size: int, workers: int, light_aug: bool, subset: int):
    if light_aug:
        # Light augmentation (baseline experiment)
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        # Heavy augmentation: helps close the lab-image vs field-image gap
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25),
        ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dir, valid_dir = data_dir / "train", data_dir / "valid"
    for d in (train_dir, valid_dir):
        if not d.is_dir():
            raise SystemExit(f"Folder not found: {d.resolve()}\n"
                             "Run from the project root and use --data .  (it needs train/ and valid/)")

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(valid_dir, val_tf)
    classes = train_ds.classes
    if classes != val_ds.classes:
        raise SystemExit("train/ and valid/ have different class folders. Fix the folders first.")
    print(f"{len(classes)} classes | {len(train_ds)} train images | {len(val_ds)} valid images")

    if subset > 0:
        train_ds = per_class_subset(train_ds, subset)
        val_ds = per_class_subset(val_ds, max(2, subset // 4))
        print(f"SMOKE TEST: using {len(train_ds)} train / {len(val_ds)} valid images")

    pin = torch.cuda.is_available()          # pin_memory only helps on GPU
    persistent = workers > 0                 # keeps workers alive between epochs (faster on Windows)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=workers,
                          pin_memory=pin, persistent_workers=persistent, drop_last=len(train_ds) > batch_size)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=workers,
                        pin_memory=pin, persistent_workers=persistent)
    return classes, train_dl, val_dl


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds += out.argmax(1).cpu().tolist()
        labels += y.tolist()
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels


def make_scaler(use_amp: bool):
    """Mixed-precision helper (only active on GPU). Works on old and new PyTorch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def export_onnx(model, path: Path):
    """Export to ONNX; batch size can vary and the input is named 'input'."""
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)   # stable exporter
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)                 # older PyTorch


def verify_onnx(model, path: Path, loader):
    """Compare ONNX output with PyTorch on real images to prove the export is correct."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x, _ = next(iter(loader))
    x = x[:8]
    with torch.no_grad():
        ref = model.eval().cpu()(x).numpy()
    out = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    same = float((ref.argmax(1) == out.argmax(1)).mean())
    print(f"ONNX check: input '{session.get_inputs()[0].name}', max diff {diff:.2e}, "
          f"same predictions {same * 100:.0f}% on {len(x)} images")
    if diff > 1e-3 or same < 1.0:
        print("WARNING: ONNX output differs from PyTorch. Do not deploy before investigating.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience")
    ap.add_argument("--workers", type=int, default=2, help="data loading processes (use 0 if Windows errors)")
    ap.add_argument("--subset", type=int, default=0, help="use only N images per class (smoke test)")
    ap.add_argument("--light-aug", action="store_true", help="light augmentation instead of heavy")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights download (testing only)")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="folder for best.pt (kept out of models/)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    args.runs.mkdir(parents=True, exist_ok=True)
    print(f"device={device}" + ("" if use_amp else "  (CPU is slow: use a GPU for the full run)"))

    classes, train_dl, val_dl = get_loaders(args.data, args.batch_size, args.workers, args.light_aug, args.subset)
    (args.out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))
    scaler = make_scaler(use_amp)

    best_acc, bad_epochs = 0.0, 0
    best_path = args.runs / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if i % 20 == 0 or i == len(train_dl):      # progress so it never looks frozen
                print(f"  epoch {epoch + 1} batch {i}/{len(train_dl)} loss={running / i:.4f}", flush=True)
        acc, _, _ = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={running / len(train_dl):.4f}  val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, bad_epochs = acc, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    # Final report with the best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    acc, preds, labels = evaluate(model, val_dl, device)
    print(f"\nBest val accuracy: {acc:.4f}\n")
    report = classification_report(labels, preds, labels=list(range(len(classes))),
                                   target_names=classes, digits=3, zero_division=0)
    print(report)
    (args.runs / "report.txt").write_text(report, encoding="utf-8")

    # Export to ONNX for lightweight inference, then verify it
    onnx_path = args.out / "model.onnx"
    export_onnx(model, onnx_path)
    verify_onnx(model, onnx_path, val_dl)
    print(f"Saved {onnx_path} and {args.out / 'classes.json'}")
    print("Next: put the val accuracy in your README table, then test on real phone photos.")


if __name__ == "__main__":   # required on Windows when using data-loading workers
    main()