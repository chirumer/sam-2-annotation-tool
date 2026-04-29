import os
import numpy as np
from PIL import Image

class _MockMaskTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)
        if self.arr.ndim == 2: self.arr = self.arr[None, :, :]
    def __gt__(self, other):
        return _MockMaskTensor((self.arr > float(other)).astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.arr.astype(bool)

class MockSAM2Predictor:
    BRUSH = 50
    def __init__(self): self._states = {}
    def init_state(self, video_path):
        def _probe(d):
            files = sorted(p for p in os.listdir(d) if p.lower().endswith(".jpg"))
            if not files: return 480, 640, 0
            img = Image.open(os.path.join(d, files[0]))
            return img.height, img.width, len(files)
        H, W, N = _probe(video_path)
        sid = "state-" + os.path.basename(os.path.normpath(video_path))
        self._states[sid] = {"H": H, "W": W, "N": N, "frames": {}}
        return sid
    def reset_state(self, state_id):
        if state_id in self._states: self._states[state_id]["frames"] = {}
    def _empty(self, state_id):
        st = self._states[state_id]
        return np.zeros((st["H"], st["W"]), dtype=bool)
    def _stamp_square(self, mask, cx, cy, value):
        H, W = mask.shape
        b = self.BRUSH // 2
        x0, x1 = int(max(0, round(cx) - b)), int(min(W, round(cx) + b))
        y0, y1 = int(max(0, round(cy) - b)), int(min(H, round(cy) + b))
        if x1 > x0 and y1 > y0: mask[y0:y1, x0:x1] = value
        return mask
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None, box=None):
        st = self._states[inference_state]
        per_obj = st["frames"].setdefault(int(frame_idx), {})
        mask = self._empty(inference_state)
        if box is not None:
            box = np.asarray(box, dtype=float).reshape(-1)
            x0, y0, x1, y1 = box
            xa, xb = int(max(0, round(min(x0, x1)))), int(min(st["W"], round(max(x0, x1))))
            ya, yb = int(max(0, round(min(y0, y1)))), int(min(st["H"], round(max(y0, y1))))
            if xb > xa and yb > ya: mask[ya:yb, xa:xb] = True
        if points is not None and len(points) > 0:
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            lbs = np.asarray(labels if labels is not None else [], dtype=int).reshape(-1)
            for (cx, cy), lab in zip(pts, lbs): self._stamp_square(mask, cx, cy, value=bool(int(lab) == 1))
        per_obj[int(obj_id)] = mask
        all_oids = sorted(per_obj.keys())
        out_logits = [_MockMaskTensor(per_obj[o].astype(np.float32) - 0.5) for o in all_oids]
        return None, all_oids, out_logits
    def propagate_in_video(self, inference_state):
        st = self._states[inference_state]
        latest = {}
        for fi in sorted(st["frames"]):
            for oid, m in st["frames"][fi].items(): latest[oid] = m
        if not latest: return
        oids = sorted(latest)
        for fi in range(st["N"]):
            yield fi, list(oids), [_MockMaskTensor(latest[o].astype(np.float32) - 0.5) for o in oids]
