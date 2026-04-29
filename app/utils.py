import os
import re
import json
import base64
import io
import zipfile
import datetime
import subprocess
import numpy as np
from PIL import Image

# Color palette: matches matplotlib tab10. JS uses the same triplets.
CMAP_RGB10 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75),  (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]

def obj_color(obj_id, alpha=0.6):
    r, g, b = CMAP_RGB10[(int(obj_id) - 1) % 10]
    return (int(r), int(g), int(b), int(round(alpha * 255)))

def mask_to_base64_png(mask, color_rgba):
    m = np.asarray(mask).astype(bool)
    if m.ndim == 3: m = m[0]
    h, w = m.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[m] = color_rgba
    pil = Image.fromarray(img, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def mask_to_rle(mask):
    """COCO uncompressed RLE (column-major) optimized with NumPy."""
    m = np.asarray(mask).astype(np.uint8)
    if m.ndim == 3: m = m[0]
    h, w = m.shape
    flat = m.T.reshape(-1)
    if flat.size == 0: return {"size": [int(h), int(w)], "counts": []}
    changes = np.where(flat[1:] != flat[:-1])[0] + 1
    idx = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(idx).tolist()
    if flat[0] == 1: counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}

def gdrive_url_to_id(url):
    if not url: return None
    url = url.strip()
    patterns = [r"/file/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})", r"open\\?id=([A-Za-z0-9_-]{10,})", r"uc\\?id=([A-Za-z0-9_-]{10,})"]
    for pat in patterns:
        m = re.search(pat, url)
        if m: return m.group(1)
    return url if re.fullmatch(r"[A-Za-z0-9_-]{10,}", url) else None

def download_gdrive(url_or_id, out_path):
    fid = gdrive_url_to_id(url_or_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = ["gdown", "--fuzzy", "-O", out_path, f"https://drive.google.com/uc?id={fid}"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path

def unzip_videos(zip_path, out_dir, exts=(".mp4", ".mov", ".mkv", ".avi", ".webm")):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf: zf.extractall(out_dir)
    found = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts: found.append(os.path.abspath(os.path.join(root, fn)))
    return sorted(found)
