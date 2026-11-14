# Motion sketch & Motion Study

Two tools for drawing the movement of people in video as hand-drawn outlines, in the spirit of
Marey's and Muybridge's chronophotography and figure-drawing motion studies.

| Tool | What it is |
|---|---|
| [`motion_study/`](motion_study/) | **Web app for time–motion study of one person.** Sketch and skeleton grids and overlaps, hands with fingers, limb contours for sitting posture, joint angles, timeline, CSV/SVG/PNG export. Runs fully in the browser (free MediaPipe models) and is published on GitHub Pages; an optional local Python server adds YOLOv8 silhouettes. See [motion_study/README.md](motion_study/README.md). |
| `motion_sketch.py` | Command-line tool: red-pen outline trails of every person over the original video (YOLOv8-seg + ByteTrack). Documented below. |

**Live web app:** `https://<your-user>.github.io/<this-repo>/` (after GitHub Pages is enabled; see motion_study/README.md).

---

# motion-sketch

Draws a hand-drawn red outline of every person in a video every few frames and keeps
the older outlines on screen, so movement shows up as stacked sketches, like
Marey's chronophotographs traced over with a red pen.

People are segmented with YOLOv8-seg and tracked with ByteTrack, so every person keeps
their own trail. Each outline is drawn as 2–3 overlapping, slightly wobbly strokes that
run a little past where they close. The jitter for an outline is generated once, so
outlines don't flicker from frame to frame.

Outputs:

1. `<input>_sketch.mp4` is the input video with the outline trails drawn on top. It has the same resolution and fps as the input.
2. `<input>_sketch.png` is a still with **every** outline recorded during the run drawn over one background frame. The middle frame is used by default.

## Install

Python 3.9+.

**Small install (~26 MB), for a Python that already has torch, torchvision, opencv,
numpy, matplotlib and tqdm** (e.g. a conda base env). The venv reuses those packages, and
`--no-deps` skips ultralytics' training-only extras: polars alone is ~170 MB.

```bash
python -m venv --system-site-packages .venv
.venv\Scripts\activate          # Windows
pip install --no-deps ultralytics ultralytics-thop lap psutil
```

`pip check` will report that polars, nvidia-ml-py and ultralytics-platform are missing.
That's expected: inference and tracking never import them. The venv depends on the base
env, so upgrading or removing torch or opencv there affects it.

**Standalone install** (downloads everything, ~1–1.5 GB with CPU torch):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

The weights (`yolov8n-seg.pt`, ~7 MB) download automatically on first run.

**GPU:** the tool uses CUDA automatically when PyTorch can see it, and prints the device
in use. `pip install ultralytics` installs a CPU-only PyTorch on Windows, so for an NVIDIA GPU, install a
CUDA build first, using the command for your setup from <https://pytorch.org/get-started/locally/>, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Examples

Defaults: an outline every 4 frames, the last 12 per person, 3 strokes, red:

```bash
python motion_sketch.py dance.mp4
```

Long fading trails on a darkened background, with inference at 960 px wide for speed:

```bash
python motion_sketch.py dance.mp4 --trail 40 --every 3 --fade --dim-bg --resize 960
```

Live webcam with a preview window. Press `q` or `Esc` to stop, and the video and still are saved on exit:

```bash
python motion_sketch.py 0 --live --resize 640 --fade
```

## Options

Run `python motion_sketch.py --help` for the full list with defaults. The main ones:

| option | default | what it does |
|---|---|---|
| `-o / --output` | `<input>_sketch.mp4` | annotated video path |
| `--still` | `<input>_sketch.png` | still image path |
| `--still-frame N` | middle frame (last when live) | background frame for the still |
| `--no-video` | off | only write the still |
| `--model` | `yolov8n-seg.pt` | any Ultralytics `-seg` model, e.g. `yolov8s-seg.pt` for cleaner masks |
| `--conf` | 0.4 | person confidence threshold |
| `--resize W` | off | run inference at width W; outlines are mapped back to full size |
| `--device` | auto | force `cpu`, `cuda:0`, … |
| `--every N` | 4 | record an outline every N frames per person |
| `--trail K` | 12 | outlines kept per person in the video (0 = all) |
| `--fade` | off | older outlines fade out linearly |
| `--dim-bg [AMOUNT]` | off (0.35 if given bare) | darken the video so the lines stand out |
| `--passes` | 3 | overlapping strokes per outline |
| `--wobble` | 3.0 | jitter in px (roughly the spread between strokes) |
| `--overshoot` | 6.0 | how far a stroke runs past its start, in px (0 = closed loops) |
| `--color` | `0,0,255` | stroke colour as `B,G,R` or a name (`red`, `black`, `blue`, …) |
| `--thickness` | 2 | stroke width in px |
| `--simplify` | 0.002 | `approxPolyDP` epsilon as a fraction of the perimeter |
| `--smooth` | 6.0 | round off mask corners over this many px before sketching |
| `--spacing` | 4.0 | resampled point spacing along the outline, in px |
| `--min-area` | 1500 | skip masks smaller than this (full-res px) |
| `--lost-frames` | 30 | stop adding outlines for an ID unseen this long (its old outlines stay) |
| `--seed` | 0 | jitter seed. With the same seed the strokes come out the same on every run |

## How it works

1. **Segment and track.** `model.track(..., classes=[0], tracker="bytetrack.yaml", retina_masks=True)`
   returns one mask per person, with a stable ID for each.
2. **Clean the mask.** The mask is opened and then closed. Only the largest connected component is kept and its holes are filled. A 5 px Gaussian blur followed by a re-threshold smooths the edge.
3. **Extract the contour.** `findContours(RETR_EXTERNAL)` gives the boundary, which is simplified with `approxPolyDP`, scaled back to full resolution, resampled to evenly spaced points and given slightly rounded corners.
4. **Sketch.** Each pass starts at a random point and goes once round the outline, then continues a few px past its start and ends with a short flick along the tangent. Gaussian-smoothed random offsets displace the points: one fine and one slow component, plus a small shift for the whole pass. All of this is computed once, when the outline is recorded.
5. **Render.** Each outline is drawn once with anti-aliased polylines (`LINE_AA`, sub-pixel
   coordinates) into a small coverage patch. The patches are max-combined into one layer, faded if `--fade` is set, and alpha-blended over the frame. The layer is rebuilt only on frames where an outline is added or dropped.

## Notes

- If a frame has no detections, the existing trails are still drawn.
- People who leave keep their outlines. With `--trail 0` they stay for the whole video.
- `yolov8n-seg` masks are coarse. Use `--model yolov8s-seg.pt` or `yolov8m-seg.pt` for tighter silhouettes, at a cost in speed.
- The still uses every outline in the run, so on a long clip with someone standing still the outlines pile up into a dense red blob. Increase `--every` if that happens.
