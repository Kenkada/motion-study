"""Detection, single-person tracking and per-frame data extraction.

The heavy lifting (segmentation, tracking, pose) happens here. Rendering of the
four outputs happens in the browser from the vector data produced here, so the
keyframe threshold, colours and opacity can be changed live without re-running
the models.

Per analysed frame we record, for the one selected person:
    i      frame index in the source
    t      time in seconds
    area   silhouette area in source pixels (from the cleaned mask)
    box    [x0, y0, x1, y1] in source pixels
    c      closed outline, evenly resampled, [[x, y], ...] in source pixels
    kp     17 COCO keypoints [[x, y, conf], ...] or null
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # C:\Mech\Software, where motion_sketch.py and the weights live
sys.path.insert(0, str(ROOT))

import motion_sketch as ms  # noqa: E402  (reuse mask cleanup + contour extraction)

SEG_MODELS = {"fast": "yolov8n-seg.pt", "accurate": "yolov8s-seg.pt"}
POSE_MODEL = "yolov8n-pose.pt"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# One model run at a time: the app is a local single-user tool and the CPU is the bottleneck.
MODEL_LOCK = threading.Lock()


def weights_path(name: str) -> str:
    """Prefer weights next to motion_sketch.py, else let Ultralytics download them there."""
    p = ROOT / name
    return str(p)


def load_model(name: str):
    from ultralytics import YOLO

    p = Path(weights_path(name))
    if p.exists():
        return YOLO(str(p))
    # Ultralytics downloads into the CWD when given a bare name; download next to the others.
    import os

    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        return YOLO(name)
    finally:
        os.chdir(cwd)


_DEVICE = None


def device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch

        _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _DEVICE


# --------------------------------------------------------------------------- #
# Sources: a video file, or a sequence of still images
# --------------------------------------------------------------------------- #
class Source:
    kind = "video"
    width = height = count = 0
    fps = 30.0

    def read(self, idx: int) -> np.ndarray | None: ...

    def iterate(self, start: int, end: int, step: int):
        """Yield (idx, frame) for idx in range(start, end + 1, step)."""
        ...

    def close(self): ...

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "fps": self.fps,
            "duration": self.count / self.fps if self.fps else 0,
        }


class VideoSource(Source):
    kind = "video"

    def __init__(self, path: Path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError("could not open the video (unsupported codec or not a video)")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps == fps and 1 <= fps <= 240 else 30.0
        self.count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, _ = cap.read()
        cap.release()
        if not ok or self.width <= 0:
            raise ValueError("the video has no readable frames")
        self._lock = threading.Lock()
        self._cap = None

    def read(self, idx: int):
        with self._lock:
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.path)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
            ok, frame = self._cap.read()
            return frame if ok else None

    def iterate(self, start: int, end: int, step: int):
        cap = cv2.VideoCapture(self.path)
        try:
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            idx = start
            while idx <= end:
                if (idx - start) % step == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield idx, frame
                else:
                    if not cap.grab():
                        break
                idx += 1
        finally:
            cap.release()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSource(Source):
    kind = "images"

    def __init__(self, paths: list[Path], interval: float = 1.0):
        self.paths = [str(p) for p in sorted(paths, key=lambda p: p.name.lower())]
        first = self._load(0)
        if first is None:
            raise ValueError("could not read the image")
        self.height, self.width = first.shape[:2]
        self.count = len(self.paths)
        self.fps = 1.0 / interval if interval > 0 else 1.0

    def _load(self, idx: int):
        # imdecode handles non-ASCII Windows paths, which imread does not.
        data = np.fromfile(self.paths[idx], dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if self.width and (img.shape[1], img.shape[0]) != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img

    def read(self, idx: int):
        if not 0 <= idx < len(self.paths):
            return None
        return self._load(idx)

    def iterate(self, start: int, end: int, step: int):
        for idx in range(start, min(end, self.count - 1) + 1, step):
            img = self._load(idx)
            if img is not None:
                yield idx, img


def open_source(paths: list[Path], interval: float = 1.0) -> Source:
    if len(paths) == 1 and paths[0].suffix.lower() not in IMAGE_EXT:
        return VideoSource(paths[0])
    imgs = [p for p in paths if p.suffix.lower() in IMAGE_EXT]
    if not imgs:
        raise ValueError("upload one video, or one or more images")
    return ImageSource(imgs, interval)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def infer_scale(width: int, max_w: int) -> float:
    """Factor from inference resolution back to source resolution."""
    return width / max_w if max_w and width > max_w else 1.0


def shrink(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(round(w / scale)), int(round(h / scale))),
                      interpolation=cv2.INTER_AREA)


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_dist(a, b) -> float:
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))


def outline_of(mask_small: np.ndarray, scale: float, spacing: float = 4.0):
    """Cleaned mask -> (outline points in source px, area in source px) or (None, 0)."""
    m = ms.clean_mask(mask_small)
    area = float(cv2.countNonZero(m)) * scale * scale
    if area <= 0:
        return None, 0.0
    pts = ms.extract_contour(m, scale, simplify=0.0015, spacing=spacing, min_area=0, smooth=4.0)
    if pts is None:
        return None, 0.0
    return pts, area


def r1(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 1).tolist()


# --------------------------------------------------------------------------- #
# Step 3a: list the people visible in one frame
# --------------------------------------------------------------------------- #
def detect_people(source: Source, frame_idx: int, quality: str, conf: float, max_w: int) -> list[dict]:
    frame = source.read(frame_idx)
    if frame is None:
        raise ValueError(f"could not read frame {frame_idx}")
    scale = infer_scale(frame.shape[1], max_w)
    small = shrink(frame, scale)
    with MODEL_LOCK:
        model = load_model(SEG_MODELS.get(quality, SEG_MODELS["fast"]))
        res = model.predict(small, classes=[0], conf=conf, retina_masks=True,
                            device=device(), verbose=False)[0]
    people = []
    if res.boxes is None or len(res.boxes) == 0:
        return people
    boxes = res.boxes.xyxy.cpu().numpy() * scale
    confs = res.boxes.conf.cpu().numpy()
    masks = res.masks.data.cpu().numpy() if res.masks is not None else [None] * len(boxes)
    for box, c, m in zip(boxes, confs, masks):
        outline = None
        if m is not None:
            pts, _ = outline_of((m > 0.5).astype(np.uint8) * 255, scale, spacing=6.0)
            if pts is not None:
                outline = r1(pts)
        people.append({"box": r1(box), "conf": round(float(c), 3), "outline": outline})
    # Number people left to right so the indices are easy to read off the preview.
    people.sort(key=lambda p: (p["box"][0] + p["box"][2]) / 2)
    for k, p in enumerate(people, start=1):
        p["index"] = k
    return people


# --------------------------------------------------------------------------- #
# Step 3b/4: follow the chosen person and record per-frame data
# --------------------------------------------------------------------------- #
def pose_for(pose_model, small: np.ndarray, target_box, scale: float):
    res = pose_model.predict(small, conf=0.25, device=device(), verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None
    boxes = res.boxes.xyxy.cpu().numpy() * scale
    best, best_iou = None, 0.3
    for k, b in enumerate(boxes):
        v = iou(b, target_box)
        if v > best_iou:
            best, best_iou = k, v
    if best is None:
        return None
    xy = res.keypoints.xy[best].cpu().numpy() * scale
    cf = res.keypoints.conf
    cf = cf[best].cpu().numpy() if cf is not None else np.ones(len(xy))
    return [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for (x, y), c in zip(xy, cf)]


# --------------------------------------------------------------------------- #
# Hands: MediaPipe hand landmarker (21 points) on a crop around each wrist
# --------------------------------------------------------------------------- #
HAND_MODEL = "hand_landmarker.task"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")


def hand_crop(kp, side: str, off: float = 0.45, scale: float = 1.3):
    """Square crop beyond the wrist along the forearm. Same rule as handCrop() in static/engine.js."""
    el, wr = (7, 9) if side == "L" else (8, 10)
    W, E = kp[wr], kp[el]
    if W is None or W[2] < 0.2:
        return None
    sh = float(np.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1])) if kp[5][2] > 0.2 and kp[6][2] > 0.2 else 0.0
    fore = float(np.hypot(W[0] - E[0], W[1] - E[1])) if E[2] > 0.2 else 0.0
    L = fore or sh * 0.8 or 100.0
    size = max(L * scale, 48.0)
    cx, cy = W[0], W[1]
    if fore:
        cx += (W[0] - E[0]) / fore * L * off
        cy += (W[1] - E[1]) / fore * L * off
    return cx - size / 2, cy - size / 2, size, W[0], W[1]


# Crop attempts, tried in order until a hand is found: (offset along forearm, size, contrast boost)
HAND_TRIES = ((0.45, 1.3, False), (0.6, 1.8, False), (0.6, 1.8, True))


class HandFinder:
    PX = 256

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        path = ROOT / HAND_MODEL
        if not path.exists():
            import urllib.request

            urllib.request.urlretrieve(HAND_URL, path)
        self.mp = mp
        self.lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))

    def close(self):
        self.lm.close()

    def find(self, frame: np.ndarray, kp) -> list | None:
        h, w = frame.shape[:2]
        picks = []
        for side in ("L", "R"):
            best = None
            for off, sc, boost in HAND_TRIES:
                cr = hand_crop(kp, side, off, sc)
                if cr is None:
                    break
                best = self._try(frame, side, cr, boost)
                if best:
                    break
            if best:
                picks.append(best)
        if len(picks) == 2:  # both crops found the same hand: keep the closer match
            a, b = picks
            palm = np.hypot(a["p"][9][0] - a["p"][0][0], a["p"][9][1] - a["p"][0][1])
            if np.hypot(a["p"][0][0] - b["p"][0][0], a["p"][0][1] - b["p"][0][1]) < 0.5 * palm:
                picks.pop(1 if a["_d"] <= b["_d"] else 0)
        for p in picks:
            p.pop("_d")
        return picks or None

    _clahe = cv2.createCLAHE(2.5, (4, 4))

    def _try(self, frame, side, cr, boost):
        x0, y0, size, wx, wy = cr
        # Affine crop (handles crops that run off the frame edge, padded with black).
        k = self.PX / size
        M = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
        crop = cv2.warpAffine(frame, M, (self.PX, self.PX), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
        if boost:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self.lm.detect(img)
        best, bd = None, size * 0.45
        for j, lms in enumerate(res.hand_landmarks or []):
            p = [(x0 + q.x * size, y0 + q.y * size) for q in lms]
            d = float(np.hypot(p[0][0] - wx, p[0][1] - wy))
            if d < bd:
                score = res.handedness[j][0].score if res.handedness and res.handedness[j] else 0.0
                bd, best = d, {"side": side, "p": [[round(x, 1), round(y, 1)] for x, y in p],
                               "s": round(float(score), 3), "_d": d / size}
        return best


def analyse(job, source: Source, start: int, end: int, step: int, target_box,
            quality: str, conf: float, max_w: int, with_pose: bool = True, with_hands: bool = True) -> None:
    """Runs in a worker thread. Appends frame records to job.frames as it goes."""
    end = min(end, source.count - 1) if source.count else end
    total = len(range(start, end + 1, step))
    job.total = total
    scale = infer_scale(source.width, max_w)
    t0 = time.perf_counter()

    with MODEL_LOCK:
        job.message = "loading models"
        # A fresh model per job, so ByteTrack state never leaks between analyses.
        seg = load_model(SEG_MODELS.get(quality, SEG_MODELS["fast"]))
        pose = None
        if with_pose:
            job.message = "loading pose model (downloads ~6 MB on first use)"
            pose = load_model(POSE_MODEL)
        hands = None
        if with_pose and with_hands:
            job.message = "loading hand model (downloads ~8 MB on first use)"
            hands = HandFinder()
        job.message = "tracking"

        tid = None
        last_box = list(target_box)
        lost = 0
        for n, (idx, frame) in enumerate(source.iterate(start, end, step)):
            if job.cancel:
                job.message = "cancelled"
                break
            small = shrink(frame, scale)
            res = seg.track(small, persist=True, classes=[0], conf=conf, tracker="bytetrack.yaml",
                            retina_masks=True, device=device(), verbose=False)[0]
            rec = {"i": idx, "t": round(idx / source.fps, 4), "area": 0, "box": None, "c": None, "kp": None, "h": None}
            if res.boxes is not None and len(res.boxes) and res.masks is not None:
                boxes = res.boxes.xyxy.cpu().numpy() * scale
                ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(boxes)
                masks = res.masks.data.cpu().numpy()
                pick = None
                if tid is not None and tid in ids:
                    pick = ids.index(tid)
                else:
                    # First frame, or the tracker lost/re-numbered the person: re-acquire by
                    # overlap with the last known box, falling back to the nearest centre.
                    scores = [iou(b, last_box) for b in boxes]
                    k = int(np.argmax(scores))
                    if scores[k] > 0.25:
                        pick = k
                    else:
                        diag = np.hypot(last_box[2] - last_box[0], last_box[3] - last_box[1])
                        dists = [center_dist(b, last_box) for b in boxes]
                        k = int(np.argmin(dists))
                        if dists[k] < 0.6 * diag * (1 + 0.1 * lost):
                            pick = k
                    if pick is not None:
                        tid = ids[pick]
                if pick is not None:
                    box = boxes[pick]
                    raw = (masks[pick] > 0.5).astype(np.uint8) * 255
                    pts, area = outline_of(raw, scale)
                    if pts is not None:
                        rec.update(area=round(area), box=r1(box), c=r1(pts))
                        last_box = [float(v) for v in box]
                        lost = 0
                        if pose is not None:
                            rec["kp"] = pose_for(pose, small, box, scale)
                            if hands is not None and rec["kp"]:
                                rec["h"] = hands.find(frame, rec["kp"])
            if rec["c"] is None:
                lost += 1
            job.frames.append(rec)
            job.done = n + 1
            el = time.perf_counter() - t0
            job.fps = (n + 1) / el if el > 0 else 0.0
        if hands is not None:
            hands.close()
    if not job.cancel:
        job.message = "done"
