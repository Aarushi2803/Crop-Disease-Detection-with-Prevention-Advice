import random, shutil
from pathlib import Path

random.seed(42)
train, valid = Path("train"), Path("valid")
VAL_FRAC = 0.2
valid.mkdir(exist_ok=True)

for cls in sorted(p for p in train.iterdir() if p.is_dir()):
    target = valid / cls.name
    if target.exists():                       # already split, skip it
        print(f"skip {cls.name}")
        continue
    imgs = [f for f in cls.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_FRAC)
    target.mkdir(parents=True)
    for f in imgs[:n_val]:
        shutil.move(str(f), str(target / f.name))
    print(f"{cls.name}: {len(imgs) - n_val} train, {n_val} valid")