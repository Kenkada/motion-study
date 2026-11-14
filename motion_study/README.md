# Motion Study

A web app for time–motion study of one person in a video. Open a video (or a set of images),
type the index of the person to follow, and get six outputs:

| Output | What it shows |
|---|---|
| **Sketch grid** | one hand-drawn pose outline per keyframe, each cell labelled with its time. Earlier poses show faintly behind the current one. |
| **Sketch overlap** | every keyframe outline on one sheet at low opacity, so the movement builds up the way charcoal does. |
| **Skeleton grid** | a stick figure per keyframe (head, spine, limbs, hand bones) over light body shading, with joint angles. |
| **Skeleton overlap** | all keyframe skeletons on one sheet, with a swept surface between consecutive poses and joint trajectories. |
| **Hands grid** | a left/right pair of close-ups per keyframe: hand outline with fingers, wrist bend and grip aperture. |
| **Hands overlap** | all keyframe hands on one sheet, with wrist, thumb-tip and index-tip trajectories. |

The lines are black on white or white on black, and the line opacity defaults to 12% (adjustable).
The source video is never drawn into the outputs. Only the sketches are.

## The pose outline

Each sketch is the **tracked silhouette** of the person, plus:

- **Hands with fingers:** a simple hand shape (palm and five fingers) built from 21 hand
  landmarks is merged into the outline, as in a figure drawing.
- **Limb contours inside the outline:** an arm across the body, folded legs when sitting, or a
  hand in front of the torso would disappear inside a plain silhouette. Their contours, from the
  tracked joints, are drawn where they fall inside the silhouette and away from its edge.
  *Limb width* adjusts them.

## Two ways to run it

**1. In the browser (default, and what the GitHub Pages site uses).** Everything runs in the
page with Google's free MediaPipe models, and the video never leaves the computer:

- a person detector (EfficientDet-Lite2) finds people, including crouched, side-on or small ones
  that a pose detector alone misses;
- pose + silhouette run on a crop around the chosen person every frame (sharper outline);
- when the pose model loses the person, a person segmenter (selfie multiclass) supplies the
  silhouette for that frame, so the outline does not drop out (those frames have no skeleton);
- the 21-point hand landmarker runs on crops around each wrist.

The models (~37 MB in total) and the MediaPipe runtime download once from Google's and
jsDelivr's CDNs and are cached by the browser.

**2. With the local Python server (optional).** The same page, plus a choice to run the analysis
on the server with YOLOv8 segmentation + ByteTrack + YOLOv8 pose, with MediaPipe for the hands.
Use it for long videos you want to keep in the `jobs\` folder, or to compare engines.

```bash
run.cmd                                    # Windows: starts the server and opens http://127.0.0.1:8765
..\.venv\Scripts\python.exe server.py      # same, from a terminal (--host 0.0.0.0 to share on a network)
```

Without Python you can also serve the static page yourself (any static server works, ES modules
need http, not file://):

```bash
python -m http.server 8000 --directory static
```

No paid or cloud API is used anywhere.

## Using it

1. **Open** a video (MP4/WebM/MOV… whatever the browser can play) or images (sorted by file
   name; set "seconds between images"). For videos the frame rate is measured; correct it if needed.
2. **Choose the person.** Move the slider to a frame where the person is clearly visible.
   Everyone detected gets an index number (left to right). Type the index and press **Analyse**.
   Analysis starts at that frame. *Analysis options*: end time, every Nth frame, model size,
   inference width, and hands on/off.
3. **Study.** The outputs fill in live while the video is tracked. All settings on the left
   re-draw them immediately, without re-running the models.

### Keyframes (which moments are drawn)

A frame becomes a keyframe when the silhouette has changed by at least the **threshold**
(default 10%) since the previous keyframe:

- **Silhouette change** (default): `max(pixels gained, pixels lost) / area of the previous keyframe`.
  This equals the area change when the body only grows or shrinks, and the fraction of the
  body that moved when it shifts or changes pose.
- **Area only**: `|area − previous area| / previous area`, area smoothed over a few frames.

### Posture and hand measures (2D, in the image plane)

| Measure | Definition |
|---|---|
| Trunk inclination | hip-centre → shoulder-centre vs. vertical (0° = upright) |
| Neck flexion | head direction (ears/nose) vs. the trunk line |
| Elbow, knee, hip | interior joint angle (180° = straight; hip ≈ 90° when sitting upright) |
| Wrist bend | hand direction (wrist → middle knuckle) vs. forearm (0° = straight) |
| Grip aperture | thumb tip – index tip distance, in palm widths |

They appear in the keyframe table, the angle chart (one measure at a time), the skeleton grid
labels (T, N, E, W, K) and the CSV. They are projections, so a limb pointing at the camera
reads shorter or straighter than it is.

### Exports

Each output: PNG, JPEG, WebP or SVG, at 1–4× the source resolution (drawn from vector data, so
larger is genuinely sharper). **SVG** is vector, the best choice for print or editing.

**Download all (ZIP):** all six as PNG + SVG, `report.html` (figures, method, parameters,
keyframe table), `keyframes.csv` (times, area, change, angles, 17 body keypoints and 2×21 hand
landmarks per keyframe), and `frames.json` (raw tracking data for every analysed frame).

**Open a saved study:** load a `frames.json` on the first page to re-draw a study without the video.

## Publishing on GitHub Pages

The repository's workflow `.github/workflows/pages.yml` publishes `motion_study/static/` on every
push to `main`. One-time setup in the repository: **Settings → Pages → Source: GitHub Actions**.
The site is then at `https://<user>.github.io/<repo>/`. On Pages only the browser engine is
offered (there is no server there).

## Collecting test runs (optional)

If `static/config.js` has a `COLLECT_URL`, the page sends a copy of every analysis to that
address: a ZIP with the six outputs as PNG, JPEG, WebP and SVG, `report.html`, `keyframes.csv`,
`frames.json` and `run_info.json`. `run_info.json` lists the models (with URLs), the engine,
the analysis parameters, the drawing settings, the source size and frame rate, the timing, the
browser, and counts of tracked frames and keyframes. A copy is sent when an analysis ends, and
again on **Download all** if the settings have changed since. Frames where nobody was detected are
reported too, with `run_info.json` only. **The video is never sent.** Testers see a notice saying
the results are shared.

The receiver is `collector/drive_collector.gs` (with `collector/appsscript.json`), a free Google Apps Script that saves each run in a
private folder of the owner's Google Drive, both as the ZIP and unzipped. It asks only for the `drive.file` permission (files it created itself), not the whole Drive. Setup instructions are at
the top of that file. With an empty `COLLECT_URL`, nothing is sent anywhere.

## Where videos are stored

- **Browser engine (the GitHub Pages site):** nowhere. The file is read straight from the
  tester's disk into the page's memory and is gone once the tab is closed.
- **Local server engine:** `motion_study/jobs/<job id>/` on the computer running `server.py`,
  with the saved outputs in `jobs/<job id>/outputs/`. That folder is excluded from git.

## Files

| File | Role |
|---|---|
| `static/index.html`, `style.css` | the page |
| `static/app.js` | flow, live updates, controls, charts, table, exports |
| `static/engine.js` | in-browser engine: MediaPipe pose + silhouette, single-person tracking, hand crops |
| `static/contour.js` | mask → clean outline (largest component, holes filled, traced, smoothed) |
| `static/render.js` | keyframes, pose outline (hands + limb contours), hand-drawn strokes, angles, the six outputs (canvas and SVG) |
| `static/zip.js` | small ZIP writer |
| `server.py` | optional FastAPI server: uploads, frames, detection, analysis jobs |
| `analysis.py` | server engine: YOLOv8 seg + ByteTrack + pose, MediaPipe hands |

## Notes

- Hands are found with a crop around each wrist, retried larger and with more contrast if
  needed. Hands that are blurred, covered (clay, gloves) or wrapped around an object are found
  less often. The silhouette is still drawn, just without the finger shapes.
- If the tracker loses the person, it re-acquires them by overlap with, or nearness to, their last
  position. Frames where they are not found show as gaps on the timeline.
- The in-browser engine runs on the CPU (WebAssembly): on many GPUs the silhouette mask reads
  back empty. Expect about 3–5 analysed frames per second; use "every Nth frame" for long clips.
