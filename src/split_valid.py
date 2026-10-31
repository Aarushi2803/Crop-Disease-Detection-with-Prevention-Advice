import os, random, shutil
from pathlib import Path

random.seed(42)
train = Path("../train")   
valid = Path("valid") 
VAL_FRAC = 0.2

if valid.exists():
    raise SystemExit("valid/ already exists, stopping so nothing is moved twice.")

for cls in sorted(p for p in train.iterdir() if p.is_dir()):
    imgs = [f for f in cls.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_FRAC)
    (valid / cls.name).mkdir(parents=True, exist_ok=True)
    for f in imgs[:n_val]:
        shutil.move(str(f), str(valid / cls.name / f.name))
    print(f"{cls.name}: {len(imgs) - n_val} train, {n_val} valid") 

print(train.resolve())
print(train.exists()) 
