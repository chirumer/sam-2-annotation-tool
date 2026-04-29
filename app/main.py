import os
import uuid
import json
import shutil
import datetime
import traceback
import threading
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .utils import *
from .models import MockSAM2Predictor

def log_debug(msg, **kwargs):
    """Log to stdout for Colab and append to /content/workspace/out/server_debug.json"""
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    entry = {"ts": ts, "msg": msg}
    entry.update(kwargs)
    
    # Print for Colab visibility
    log_str = f"[{ts}] {msg}"
    if kwargs:
        log_str += f" | {json.dumps(kwargs)}"
    print(log_str, flush=True)

    try:
        out_dir = "/content/workspace/out"
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "server_debug.json")
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

@dataclass
class AnnotationSession:
    predictor: Any = None
    device: Any = None
    inference_state: Any = None
    video_path: Optional[str] = None
    preview_path: Optional[str] = None
    frames_dir: Optional[str] = None
    frame_names: list = field(default_factory=list)
    fps: float = 0.0
    width: int = 0
    height: int = 0
    n_frames: int = 0
    prompts: dict = field(default_factory=dict)
    video_segments: dict = field(default_factory=dict)
    last_propagated: bool = False
    events: list = field(default_factory=list)
    output_dir: str = "/content/workspace/out"
    source_meta: dict = field(default_factory=dict)
    video_registry: dict = field(default_factory=dict)
    use_mock: bool = False

    def reset_for_new_video(self):
        self.prompts.clear()
        self.video_segments.clear()
        self.events.clear()
        self.last_propagated = False

    def log(self, kind, **fields):
        ev = {"ts": datetime.datetime.utcnow().isoformat() + "Z", "kind": kind}
        ev.update(fields)
        self.events.append(ev)

SESSION = AnnotationSession()
app = FastAPI(title="SAM 2 Annotator API")

def build_annotations_json(session):
    stem = os.path.splitext(os.path.basename(session.video_path or "video"))[0]
    all_obj_ids = sorted({oid for m in session.video_segments.values() for oid in m})
    objects = []
    for oid in all_obj_ids:
        history = []
        for fi, p in sorted(session.prompts.get(oid, {}).items()):
            history.append({
                "frame_idx": int(fi),
                "points":    p.get("points", []),
                "labels":    p.get("labels", []),
                "box":       p.get("box"),
            })
        objects.append({
            "obj_id":     int(oid),
            "color_rgba": list(obj_color(int(oid))),
            "prompts":    history,
        })
    frames_json = {}
    for fi, per_obj in sorted(session.video_segments.items()):
        frames_json[str(int(fi))] = {
            str(int(oid)): mask_to_rle(mask) for oid, mask in per_obj.items()
        }
    return {
        "schema_version": "1.0",
        "video": {
            "name":     os.path.basename(session.video_path or ""),
            "stem":     stem,
            "fps":      float(session.fps),
            "width":    int(session.width),
            "height":   int(session.height),
            "n_frames": int(session.n_frames),
            "source":   session.source_meta,
        },
        "model": {
            "checkpoint": "sam2.1_hiera_large.pt",
            "config":     "configs/sam2.1/sam2.1_hiera_l.yaml",
            "device":     str(session.device),
            "predictor":  "SAM2VideoPredictor",
        },
        "objects": objects,
        "frames":  frames_json,
        "events":  session.events,
    }

def set_mock_mode(enabled: bool, device="cpu"):
    SESSION.use_mock = enabled
    SESSION.device = device
    if enabled:
        SESSION.predictor = MockSAM2Predictor()

# Request Models
class AddPromptReq(BaseModel):
    frame_idx: int
    obj_id: int
    points: List[List[float]] = []
    labels: List[int] = []
    box: Optional[List[float]] = None

class ClearFrameReq(BaseModel):
    frame_idx: int
    obj_id: int

class DriveReq(BaseModel):
    url: str

class InitVideoReq(BaseModel):
    stem: str

@app.get("/", response_class=HTMLResponse)
def index():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(static_path, "r") as f: return f.read()

@app.get("/meta")
def meta():
    try:
        return {
            "ready": SESSION.inference_state is not None,
            "fps": float(SESSION.fps),
            "width": int(SESSION.width),
            "height": int(SESSION.height),
            "n_frames": int(SESSION.n_frames),
            "video_name": os.path.basename(SESSION.video_path or ""),
            "source": SESSION.source_meta,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/videos")
def videos():
    try:
        current = next((s for s, r in SESSION.video_registry.items() if SESSION.video_path and os.path.abspath(SESSION.video_path) == os.path.abspath(r["video_path"])), None)
        return {"ok": True, "current": current, "videos": [{"stem": s, "name": os.path.basename(r["video_path"]), **{k: r[k] for k in ["width", "height", "n_frames", "source_type"]}} for s, r in sorted(SESSION.video_registry.items())]}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/upload_video")
async def upload_video(files: List[UploadFile] = File(...)):
    try:
        raw_dir = "/content/workspace/raw"
        os.makedirs(raw_dir, exist_ok=True)
        paths = []
        for upload in files:
            dest = os.path.join(raw_dir, f"{uuid.uuid4().hex[:8]}_{upload.filename}")
            with open(dest, "wb") as f: f.write(await upload.read())
            if dest.lower().endswith(".zip"): paths.extend(unzip_videos(dest, os.path.join(raw_dir, uuid.uuid4().hex[:8])))
            else: paths.append(dest)
        return register_videos_for_server(paths, "local_upload", "")
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/init_video")
def init_video(req: InitVideoReq):
    try:
        rec = SESSION.video_registry[req.stem]
        if SESSION.inference_state: SESSION.predictor.reset_state(SESSION.inference_state)
        SESSION.inference_state = SESSION.predictor.init_state(video_path=rec["frames_dir"])
        SESSION.video_path = rec["video_path"]
        SESSION.frames_dir = rec["frames_dir"]
        SESSION.preview_path = rec["preview_path"]
        SESSION.fps, SESSION.width, SESSION.height, SESSION.n_frames = rec["fps"], rec["width"], rec["height"], rec["n_frames"]
        SESSION.source_meta = {"type": rec["source_type"], "url_or_filename": rec["source_url"] or os.path.basename(rec["video_path"])}
        SESSION.reset_for_new_video()
        return {"ok": True, "video": {"name": os.path.basename(rec["video_path"])}}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/video.mp4")
def video_mp4():
    if not SESSION.preview_path or not os.path.exists(SESSION.preview_path): raise HTTPException(404)
    return FileResponse(SESSION.preview_path, media_type="video/mp4")

@app.post("/add_prompt")
def add_prompt(req: AddPromptReq):
    try:
        if not SESSION.inference_state: raise RuntimeError("No video initialized")
        fi, oid = req.frame_idx, req.obj_id
        pts = np.asarray(req.points, dtype=np.float32).reshape(-1, 2) if req.points else np.zeros((0, 2), np.float32)
        lbs = np.asarray(req.labels, dtype=np.int32) if req.labels else np.zeros((0,), np.int32)
        box = np.asarray(req.box, dtype=np.float32) if req.box else None
        SESSION.prompts.setdefault(oid, {})[fi] = {"points": pts.tolist(), "labels": lbs.tolist(), "box": box.tolist() if box is not None else None}
        kw = {"inference_state": SESSION.inference_state, "frame_idx": fi, "obj_id": oid}
        if box is not None: kw["box"] = box
        if len(pts): kw["points"], kw["labels"] = pts, lbs
        _, out_obj_ids, out_logits = SESSION.predictor.add_new_points_or_box(**kw)
        masks = {}
        for i, oo in enumerate(out_obj_ids):
            m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
            SESSION.video_segments.setdefault(fi, {})[int(oo)] = m
            masks[str(int(oo))] = mask_to_base64_png(m, obj_color(int(oo)))
        return {"ok": True, "masks": masks}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/propagate")
def propagate():
    try:
        if not SESSION.inference_state: raise RuntimeError("No video initialized")
        out_frames = {}
        for fi, obj_ids, logits in SESSION.predictor.propagate_in_video(SESSION.inference_state):
            per = {}
            for i, oo in enumerate(obj_ids):
                m = (logits[i] > 0.0).cpu().numpy().squeeze()
                SESSION.video_segments.setdefault(int(fi), {})[int(oo)] = m
                per[str(int(oo))] = mask_to_base64_png(m, obj_color(int(oo)))
            out_frames[str(int(fi))] = per
        return {"ok": True, "frames": out_frames, "n_frames": len(out_frames)}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/done")
def done():
    try:
        log_debug("Done action triggered. Persisting annotations to content/workspace/out...")
        data = build_annotations_json(SESSION)
        stem = data.get("video", {}).get("stem") or "video"
        out_dir = os.path.join(SESSION.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "annotations.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        log_debug(f"Successfully saved annotations to {out_path}")
        return {"ok": True, "saved_path": out_path, "n_objects": len(data.get("objects", []))}
    except Exception as e:
        tb = traceback.format_exc()
        log_debug("done failed", error=str(e), traceback=tb)
        return {"ok": False, "error": str(e), "traceback": tb}

@app.get("/annotations.json")
def annotations():
    try:
        return build_annotations_json(SESSION)
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": traceback.format_exc()})

def register_videos_for_server(paths, source_type, source_url):
    for vp in paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        fdir = os.path.join("/content/workspace/frames", stem)
        if os.path.exists(fdir): shutil.rmtree(fdir)
        from .utils import probe_video, extract_frames_from_video, make_preview_mp4
        info = extract_frames_from_video(vp, fdir)
        preview = make_preview_mp4(vp, os.path.join("/content/workspace/previews", stem + ".mp4"))
        SESSION.video_registry[stem] = {"video_path": os.path.abspath(vp), "frames_dir": fdir, "preview_path": preview, "source_type": source_type, "source_url": source_url, **info}
    return {"ok": True}
