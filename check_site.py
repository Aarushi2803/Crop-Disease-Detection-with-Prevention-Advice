"""CropDoc site check. Run from the PROJECT ROOT while uvicorn is running in another window:

    python check_site.py

Paste the whole output back to get help. Uses only the Python standard library.
"""
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
root = Path.cwd()
print("Folder:", root)

# ---- 1. Are we in the right folder, and are the files there?
print("\n[1] Files")
for rel in ["app/main.py", "app/static/index.html", "app/static/style.css", "app/static/script.js",
            "models/model.onnx", "models/classes.json", "data/prevention.json"]:
    p = root / rel
    print(f"  {'OK     ' if p.exists() else 'MISSING'} {rel}" + (f"  ({p.stat().st_size} bytes)" if p.exists() else ""))

static = root / "app" / "static"
if static.is_dir():
    print("  Actual names in app/static:", sorted(x.name for x in static.iterdir()))
    print("  (names must be exactly: index.html, script.js, style.css, all lowercase)")

# ---- 2. What do the links in index.html say?
print("\n[2] Links inside app/static/index.html")
idx = static / "index.html"
if idx.exists():
    for n, line in enumerate(idx.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(r"style\.css|script\.js", line, re.I):
            print(f"  line {n}: {line.strip()}")
    print("  Correct links are: /static/style.css and /static/script.js")


# ---- 3. What does the running server answer?
def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return None, str(e)


print("\n[3] Running server at", BASE)
code, body = get("/api/health")
if code is None:
    print("  CANNOT CONNECT. uvicorn is not running (or crashed). Start it in its own window:")
    print('    cd "D:\\Crop Disease Detection"')
    print("    uvicorn app.main:app --reload")
    sys.exit(0)
print(f"  /api/health -> {code} {body[:150]}")

code, body = get("/")
print(f"  /           -> {code}")
for line in body.splitlines():
    if re.search(r"style\.css|script\.js", line, re.I):
        print("     page link:", line.strip())
for path in ("/static/style.css", "/static/script.js"):
    code, _ = get(path)
    print(f"  {path} -> {code}  ({'OK' if code == 200 else 'PROBLEM'})")

print("\nDone. Paste everything above into the chat.")