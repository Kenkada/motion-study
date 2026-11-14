#!/usr/bin/env python3
"""motion_sketch.py - stacked red-pen outlines of moving people.

Segments every person in a video with YOLOv8-seg, tracks them with ByteTrack,
and every few frames records a hand-drawn-looking outline of each silhouette.
Older outlines stay on screen, so the output reads like chronophotography
traced over with a red pen.

Outputs an annotated video (same size and fps as the input) and a still PNG
with every outline ever recorded drawn over one background frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PERSON_CLASS = 0
NAMED_COLORS = {  # BGR
    "red": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue": (255, 0, 0),
    "green": (0, 160, 0),
    "yellow": (0, 220, 255),
    "orange": (0, 128, 255),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Outline:
    frame_idx: int
    strokes: list[np.ndarray]  # each (M, 2) float32, full-resolution pixel coords
    box: tuple[int, int, int, int] | None = None  # cached raster patch location
    patch: np.ndarray | None = None  # cached uint8 coverage, freed when it leaves the trail


@dataclass
class Track:
    tid: int
    trail: deque  # most recent outlines, used for the video
    history: list = field(default_factory=list)  # every outline, used for the still
    last_seen: int = -1
    last_recorded: int = -10**9
    retired: bool = False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_color(text: str) -> tuple[int, int, int]:
    key = text.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    try:
        parts = [int(v) for v in key.replace(" ", "").split(",")]
    except ValueError:
        parts = []
    if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
        raise argparse.ArgumentTypeError(
            f"colour must be 'B,G,R' (0-255 each) or one of {', '.join(NAMED_COLORS)}"
        )
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Draw stacked hand-drawn outlines of moving people (motion sketch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="video file (mp4/mov/...), or a webcam index with --live")
    p.add_argument("--live", action="store_true",
                   help="read from a webcam (input is the camera index, default 0) and show a preview window")
    p.add_argument("-o", "--output", default=None,
                   help="annotated video path (default: <input>_sketch.mp4)")
    p.add_argument("--still", default=None,
                   help="still PNG path (default: <input>_sketch.png)")
    p.add_argument("--still-frame", type=int, default=None,
                   help="frame index used as the still's background (default: middle frame; last frame when live)")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")

    g = p.add_argument_group("segmentation and tracking")
    g.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics segmentation weights")
    g.add_argument("--conf", type=float, default=0.4, help="person confidence threshold")
    g.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config")
    g.add_argument("--resize", type=int, default=None, metavar="W",
                   help="downscale frames to width W for inference (outlines are mapped back to full size)")
    g.add_argument("--device", default=None, help="force a device, e.g. cpu, cuda:0 (default: auto)")
    g.add_argument("--min-area", type=float, default=1500,
                   help="ignore person masks smaller than this many full-resolution pixels")
    g.add_argument("--lost-frames", type=int, default=30,
                   help="stop adding outlines for a track ID unseen for more than this many frames")

    g = p.add_argument_group("outline")
    g.add_argument("--every", type=int, default=4, help="record an outline every N frames per person")
    g.add_argument("--simplify", type=float, default=0.002,
                   help="approxPolyDP epsilon as a fraction of the contour perimeter")
    g.add_argument("--spacing", type=float, default=4.0,
                   help="resampled point spacing along the outline, in full-resolution pixels")
    g.add_argument("--smooth", type=float, default=6.0,
                   help="round off mask corners over this many pixels before sketching (0 = off)")

    g = p.add_argument_group("hand-drawn look")
    g.add_argument("--passes", type=int, default=3, help="overlapping strokes per outline")
    g.add_argument("--wobble", type=float, default=3.0, help="jitter amplitude in pixels")
    g.add_argument("--overshoot", type=float, default=6.0,
                   help="how far strokes run past the closure point, in pixels (0 = off)")
    g.add_argument("--color", type=parse_color, default=(0, 0, 255),
                   help="stroke colour as 'B,G,R' or a name (red, black, blue, ...)")
    g.add_argument("--thickness", type=int, default=2, help="stroke thickness in pixels")
    g.add_argument("--seed", type=int, default=0, help="random seed for the stroke jitter")

    g = p.add_argument_group("trails and background")
    g.add_argument("--trail", type=int, default=12,
                   help="outlines kept per person in the video (0 = keep all)")
    g.add_argument("--fade", action="store_true", help="older outlines fade out linearly")
    g.add_argument("--dim-bg", type=float, nargs="?", const=0.35, default=0.0, metavar="AMOUNT",
                   help="darken the video by AMOUNT (0-1); bare flag uses 0.35")
    return p


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.every < 1:
        parser.error("--every must be >= 1")
    if args.passes < 1:
        parser.error("--passes must be >= 1")
    if args.trail < 0:
        parser.error("--trail must be >= 0")
    if args.thickness < 1:
        parser.error("--thickness must be >= 1")
    if not 0.0 <= args.dim_bg < 1.0:
        parser.error("--dim-bg must be in [0, 1)")
    if not 0.0 < args.conf < 1.0:
        parser.error("--conf must be in (0, 1)")
    if args.smooth < 0:
        parser.error("--smooth must be >= 0")
    if args.spacing <= 0:
        parser.error("--spacing must be > 0")
    if args.resize is not None and args.resize < 32:
        parser.error("--resize must be >= 32")
    if args.live:
        try:
            args.source = int(args.input) if args.input is not None else 0
        except ValueError:
            parser.error("with --live, input must be a webcam index such as 0")
        stem = Path(f"motion_sketch_cam{args.source}")
    else:
        if args.input is None:
            parser.error("an input video is required (or use --live)")
        if not Path(args.input).is_file():
            parser.error(f"input not found: {args.input}")
        args.source = args.input
        src = Path(args.input)
        stem = src.with_name(src.stem)
    args.output = args.output or f"{stem}_sketch.mp4"
    args.still = args.still or f"{stem}_sketch.png"


def pick_device(forced: str | None) -> str:
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def describe_device(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        idx = int(device.split(":")[1]) if ":" in device else 0
        return f"{device} ({torch.cuda.get_device_name(idx)})"
    return device


# --------------------------------------------------------------------------- #
# 1. Segmentation
# --------------------------------------------------------------------------- #
def segment(model, frame: np.ndarray, args, device: str) -> list[tuple[int, np.ndarray]]:
    """Run tracked person segmentation. Returns [(track_id, uint8 mask 0/255)].

    Masks are at the resolution of `frame` (retina_masks=True), so callers that
    downscale before inference get downscaled masks back.
    """
    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS],
        conf=args.conf,
        tracker=args.tracker,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    res = results[0]
    if res.masks is None or res.boxes is None or res.boxes.id is None:
        return []
    ids = res.boxes.id.int().cpu().tolist()
    masks = res.masks.data.cpu().numpy()
    h, w = frame.shape[:2]
    out = []
    for tid, m in zip(ids, masks):
        if m.shape != (h, w):  # defensive: retina masks should already match
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append((tid, (m > 0.5).astype(np.uint8) * 255))
    return out


# --------------------------------------------------------------------------- #
# 2. Mask cleanup
# --------------------------------------------------------------------------- #
def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Open, close, keep largest component, fill holes, blur, re-threshold."""
    h, w = mask.shape
    k = max(3, int(round(min(h, w) / 160)) | 1)  # ~5 px at 640-ish, odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill holes: redraw the outer boundary filled.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, contours, -1, 255, thickness=cv2.FILLED)

    m = cv2.GaussianBlur(m, (5, 5), 0)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


# --------------------------------------------------------------------------- #
# 3. Contour extraction
# --------------------------------------------------------------------------- #
def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly space points along a closed polyline."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = cum[-1]
    n = max(24, int(round(perimeter / spacing)))
    t = np.linspace(0.0, perimeter, n, endpoint=False)
    x = np.interp(t, cum, closed[:, 0])
    y = np.interp(t, cum, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def round_corners(points: np.ndarray, radius_px: float, spacing: float) -> np.ndarray:
    """Circular Gaussian smoothing of an evenly spaced closed outline.

    Softens the straight runs and sharp corners left by the low-res mask and
    approxPolyDP, so the outline reads as a drawn curve rather than a polygon.
    """
    sigma = radius_px / spacing
    if sigma < 0.5 or len(points) < 8:
        return points
    r = min(int(3 * sigma), len(points) // 2 - 1)
    x = np.arange(-r, r + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([points[-r:], points, points[:r]])
    sm = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)
    return sm.astype(np.float32)


def extract_contour(mask: np.ndarray, scale: float, simplify: float, spacing: float,
                    min_area: float, smooth: float = 0.0) -> np.ndarray | None:
    """Largest external contour, simplified, mapped to full resolution, resampled."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) * scale * scale < min_area:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, simplify * peri, True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return None
    # Pixel centres: map (x + 0.5) * scale - 0.5 so downscaled coords line up.
    full = (approx + 0.5) * scale - 0.5
    return round_corners(resample_closed(full, spacing), smooth, spacing)


# --------------------------------------------------------------------------- #
# 4/5. Hand-drawn strokes
# --------------------------------------------------------------------------- #
def smooth_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """(n, 2) Gaussian-smoothed random walk offsets with unit std per axis."""
    sigma = max(1.0, sigma)
    radius = int(3 * sigma)
    raw = rng.standard_normal((n + 2 * radius, 2))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    sm = np.stack([np.convolve(raw[:, i], kernel, mode="valid") for i in range(2)], axis=1)
    sm = sm[:n]
    std = sm.std(axis=0)
    std[std < 1e-6] = 1.0
    return (sm - sm.mean(axis=0)) / std


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


def sketchify(points: np.ndarray, rng: np.random.Generator, passes: int, wobble: float,
              overshoot: float, spacing: float) -> list[np.ndarray]:
    """Turn one clean closed outline into `passes` wobbly, overshooting open strokes.

    Jitter is generated once here and stored, so a recorded outline looks the
    same on every frame it is drawn (no flicker).
    """
    n = len(points)
    seg = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    perimeter = float(seg.sum())
    # Wobble wavelength scales with the figure so small and large people look alike.
    sigma_pts = max(10.0, perimeter / 80.0) / spacing
    strokes = []
    for p in range(passes):
        start = int(rng.integers(n))
        extra = 0
        if overshoot > 0:
            extra = int(round(overshoot * rng.uniform(1.0, 2.5) / spacing))
        idx = (start + np.arange(n + extra)) % n
        path = points[idx].astype(np.float64)

        if overshoot > 0 and len(path) > 3:
            # Pen lead-in and flick-off along the local tangent.
            head_dir = _unit(path[0] - path[2])
            tail_dir = _unit(path[-1] - path[-3])
            lead = path[0] + head_dir * overshoot * rng.uniform(0.2, 0.8)
            tail = path[-1] + tail_dir * overshoot * rng.uniform(0.5, 1.2)
            path = np.vstack([lead, path, tail])

        amp = wobble * (0.6 if p == 0 else 0.9)  # std of the jitter, in px
        # Noise runs along the open path (not around the loop), so the overlap
        # at the closure point lands slightly off the start, like a real pen.
        noise = smooth_noise(rng, len(path), sigma_pts) * amp
        noise += 0.7 * smooth_noise(rng, len(path), sigma_pts * 4) * amp  # slower drift
        drift = rng.normal(0.0, wobble * 0.5, size=2)  # each pass sits slightly off the last
        strokes.append((path + noise + drift).astype(np.float32))
    return strokes


# --------------------------------------------------------------------------- #
# 6/7. Rendering
# --------------------------------------------------------------------------- #
_SHIFT = 4
_SUB = 1 << _SHIFT


def _draw_strokes(canvas: np.ndarray, strokes: list[np.ndarray], thickness: int,
                  offset: tuple[int, int] = (0, 0)) -> None:
    ox, oy = offset
    for i, s in enumerate(strokes):
        pts = np.round((s - (ox, oy)) * _SUB).astype(np.int32).reshape(-1, 1, 2)
        t = thickness if i == 0 else max(1, thickness - (1 if thickness > 2 else 0))
        cv2.polylines(canvas, [pts], False, 255, t, cv2.LINE_AA, _SHIFT)


Box = tuple[int, int, int, int]  # y0, y1, x0, x1


def rasterize(outline: Outline, shape: tuple[int, int], thickness: int,
              cache: bool = True) -> tuple[Box | None, np.ndarray | None]:
    """Draw an outline's strokes into a tight uint8 coverage patch.

    Thick anti-aliased polylines are the slowest part of rendering, so each
    outline is drawn once and the patch is reused while it stays in a trail.
    """
    if outline.patch is not None:
        return outline.box, outline.patch
    h, w = shape
    pad = thickness + 2
    allpts = np.vstack(outline.strokes)
    x0 = max(0, int(np.floor(allpts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(allpts[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(allpts[:, 0].max())) + pad + 1)
    y1 = min(h, int(np.ceil(allpts[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None
    patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _draw_strokes(patch, outline.strokes, thickness, (x0, y0))
    box = (y0, y1, x0, x1)
    if cache:
        outline.box, outline.patch = box, patch
    return box, patch


def union_box(a: Box | None, b: Box | None) -> Box | None:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def render_alpha(shape: tuple[int, int], outlines: list[tuple[Outline, float]], thickness: int,
                 alpha: np.ndarray | None = None, box: Box | None = None,
                 cache: bool = True) -> tuple[np.ndarray, Box | None]:
    """Max-composite (outline, opacity) pairs into a coverage layer.

    Pass an existing `alpha`/`box` to add outlines on top of it.
    """
    if alpha is None:
        alpha = np.zeros(shape, np.uint8)
    for outline, opacity in outlines:
        if opacity <= 0.0:
            continue
        b, patch = rasterize(outline, shape, thickness, cache)
        if b is None:
            continue
        if opacity < 0.999:
            patch = cv2.convertScaleAbs(patch, alpha=opacity)
        y0, y1, x0, x1 = b
        roi = alpha[y0:y1, x0:x1]
        np.maximum(roi, patch, out=roi)
        box = union_box(box, b)
    return alpha, box


class Overlay:
    """Coverage layer prepared for blending: bounding box plus float weights."""

    def __init__(self, alpha: np.ndarray, box: Box | None, color: tuple[int, int, int]):
        self.box = box
        if box is None:
            return
        y0, y1, x0, x1 = box
        w = alpha[y0:y1, x0:x1].astype(np.float32) * (1.0 / 255.0)
        self.w_ink = w
        self.w_bg = 1.0 - w
        self.ink = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
        self.ink[:] = color


def render(frame: np.ndarray, overlay: Overlay, dim: float) -> np.ndarray:
    """Dim the frame (optional) and alpha-blend the ink layer over it."""
    # Both branches return a new array, so the input frame is never modified.
    out = frame.copy() if dim <= 0 else cv2.convertScaleAbs(frame, alpha=1.0 - dim)
    if overlay.box is None:
        return out
    y0, y1, x0, x1 = overlay.box
    out[y0:y1, x0:x1] = cv2.blendLinear(out[y0:y1, x0:x1], overlay.ink,
                                        overlay.w_bg, overlay.w_ink)
    return out


def trail_outlines(tracks: dict[int, Track], fade: bool) -> list[tuple[Outline, float]]:
    items = []
    for tr in tracks.values():
        k = len(tr.trail)
        for i, o in enumerate(tr.trail):  # oldest first
            op = (i + 1) / k if fade else 1.0
            items.append((o, op))
    return items


class TrailRenderer:
    """Keeps the trail layer between frames and updates it only when outlines change.

    Without --fade, new outlines are simply added on top; the layer is rebuilt
    from cached patches only when an old outline leaves a trail. With --fade
    every opacity shifts when a trail grows, so the layer is rebuilt.
    """

    def __init__(self, shape: tuple[int, int], args):
        self.shape = shape
        self.args = args
        self.key = None
        self.alpha = np.zeros(shape, np.uint8)
        self.box: Box | None = None
        self.drawn: set[int] = set()
        self.overlay = Overlay(self.alpha, None, args.color)

    def __call__(self, frame: np.ndarray, tracks: dict[int, Track]) -> np.ndarray:
        key = tuple((tid, len(tr.history)) for tid, tr in tracks.items())
        if key != self.key:
            self.key = key
            items = trail_outlines(tracks, self.args.fade)
            current = {id(o) for o, _ in items}
            if not self.args.fade and self.drawn <= current:
                new = [(o, op) for o, op in items if id(o) not in self.drawn]
                self.alpha, self.box = render_alpha(self.shape, new, self.args.thickness,
                                                    self.alpha, self.box)
            else:
                self.alpha, self.box = render_alpha(self.shape, items, self.args.thickness)
            self.drawn = current
            self.overlay = Overlay(self.alpha, self.box, self.args.color)
        return render(frame, self.overlay, self.args.dim_bg)


def render_still(background: np.ndarray, tracks: dict[int, Track], args) -> np.ndarray:
    """Every outline ever recorded, at full opacity, over one background frame."""
    every = [(o, 1.0) for tr in tracks.values() for o in tr.history]
    alpha, box = render_alpha(background.shape[:2], every, args.thickness, cache=False)
    return render(background, Overlay(alpha, box, args.color), args.dim_bg)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def update_tracks(tracks: dict[int, Track], detections, frame_idx: int, scale: float,
                  args, rng: np.random.Generator) -> None:
    maxlen = args.trail if args.trail > 0 else None
    for tid, raw in detections:
        tr = tracks.get(tid)
        if tr is None:
            tr = tracks[tid] = Track(tid=tid, trail=deque(maxlen=maxlen))
        elif tr.retired:
            continue
        tr.last_seen = frame_idx
        if frame_idx - tr.last_recorded < args.every:
            continue
        pts = extract_contour(clean_mask(raw), scale, args.simplify, args.spacing,
                              args.min_area, args.smooth)
        if pts is None:
            continue
        strokes = sketchify(pts, rng, args.passes, args.wobble, args.overshoot, args.spacing)
        outline = Outline(frame_idx, strokes)
        if tr.trail.maxlen is not None and len(tr.trail) == tr.trail.maxlen:
            tr.trail[0].patch = None  # about to drop out of the trail; free its raster
        tr.trail.append(outline)
        tr.history.append(outline)
        tr.last_recorded = frame_idx

    for tr in tracks.values():
        if not tr.retired and tr.last_seen >= 0 and frame_idx - tr.last_seen > args.lost_frames:
            tr.retired = True


def open_capture(args) -> cv2.VideoCapture:
    if args.live and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"error: could not open {'camera' if args.live else 'video'} {args.source!r}")
    return cap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)

    cap = open_capture(args)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 30.0
    total = 0 if args.live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 0:
        total = 0

    still_idx = args.still_frame
    if still_idx is None and total > 0:
        still_idx = total // 2

    infer_w = args.resize if args.resize and args.resize < width else width
    infer_h = int(round(height * infer_w / width)) if infer_w != width else height
    scale = width / infer_w

    device = pick_device(args.device)
    print(f"device: {describe_device(device)}")
    print(f"input: {args.source} {width}x{height} @ {fps:.2f} fps"
          + (f", {total} frames" if total else "")
          + (f"; inference at {infer_w}x{infer_h}" if infer_w != width else ""))

    from ultralytics import YOLO  # imported late so --help stays fast

    model = YOLO(args.model)
    if getattr(model, "task", None) != "segment":
        sys.exit(f"error: {args.model} is not a segmentation model (task={model.task})")

    writer = None
    if not args.no_video:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            sys.exit(f"error: could not open video writer for {args.output}")

    rng = np.random.default_rng(args.seed)
    tracks: dict[int, Track] = {}
    trail = TrailRenderer((height, width), args)
    still_bg = None
    last_frame = None
    frame_idx = 0
    infer_time = 0.0
    t_start = time.perf_counter()
    bar = tqdm(total=total or None, unit="frame", dynamic_ncols=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            small = frame if infer_w == width else cv2.resize(
                frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            detections = segment(model, small, args, device)
            infer_time += time.perf_counter() - t0
            update_tracks(tracks, detections, frame_idx, scale, args, rng)

            if still_idx is not None and frame_idx == still_idx:
                still_bg = frame.copy()
            last_frame = frame

            out = trail(frame, tracks)
            if writer is not None:
                writer.write(out)
            if args.live:
                cv2.imshow("motion sketch (q to quit)", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_idx += 1
            bar.update(1)
            bar.set_postfix(people=sum(not t.retired for t in tracks.values()),
                            outlines=sum(len(t.history) for t in tracks.values()))
    except KeyboardInterrupt:
        print("\ninterrupted - writing what we have")
    finally:
        bar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.live:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    if frame_idx == 0:
        sys.exit("error: no frames were read")

    if still_bg is None:
        if still_idx is not None and still_idx != frame_idx - 1:
            print(f"note: still frame {still_idx} not reached; using last frame {frame_idx - 1}")
        still_bg = last_frame
    still = render_still(still_bg, tracks, args)
    Path(args.still).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.still, still)

    n_outlines = sum(len(t.history) for t in tracks.values())
    print(f"frames: {frame_idx}  people tracked: {len(tracks)}  outlines: {n_outlines}")
    print(f"average fps: {frame_idx / elapsed:.1f} overall, "
          f"{frame_idx / max(infer_time, 1e-9):.1f} inference only")
    if writer is not None:
        print(f"video: {args.output}")
    print(f"still: {args.still}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
