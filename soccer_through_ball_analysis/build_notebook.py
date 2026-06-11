"""Generates soccer_through_ball_analysis.ipynb.

Run:  python3 build_notebook.py
Produces a Colab-ready notebook for through-ball CV analysis with Supervision.
The cell sources live here as plain strings so they are easy to edit/diff;
re-run this script to regenerate the .ipynb after any change.
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "soccer_through_ball_analysis.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append(
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}
    )


# ---------------------------------------------------------------------------
# CELL 0 — README (markdown)
# ---------------------------------------------------------------------------
md(r'''# ⚽ Through-Ball Analysis — Barcelona 2010/11 (Supervision + YOLOv8)

Turns short broadcast clips of **through-ball sequences** into coach-style
playbook breakdowns: tracked players, ball, run traces, and freeze-frames with
pass / run arrows and a highlighted space zone. Record a voiceover on top in
CapCut and you have the analysis video.

## How to use this notebook (run cells top to bottom)

| Cell | What it does | You do |
|---|---|---|
| **1 Setup** | Installs libs, mounts Drive, makes output folders | Just run it (~3 min) |
| **2 Download + trim** | yt-dlp search → raw match video; ffmpeg trim → clip; **preview frames** | Confirm the window before trimming |
| **3 Detect + track** | Players + ball, ByteTrack IDs, annotated MP4, **caches coords** | Run on ONE clip first |
| **4 Read IDs** | Contact sheet of tracked frames so you can read passer/receiver IDs | Note the two ID numbers |
| **5 Arrows** | Freeze-frame PNGs: pass arrow + run arrow + space zone | Give 2 frame numbers + 2 IDs |
| **6 Batch** | Runs cells 3 on the other clips | Run after clip #1 looks clean |
| **7 Export** | Lists everything saved to Drive | Run, then download for CapCut |

## Where outputs land (Google Drive)

```
MyDrive/soccer_through_ball/
├── clips/          # trimmed source clips (clip_*.mp4)
├── outputs/        # tracked videos (*_tracked.mp4) + coord caches (*.pkl)
└── freezeframes/   # playbook PNGs (*_playbook_*.png) + ID contact sheets
```
Everything saves to Drive the moment it is produced — a dead Colab session loses nothing.

## Read this before you start (the honest caveats)

1. **Ball detection on broadcast footage is unreliable.** The ball is tiny, fast,
   motion-blurred. YOLOv8x's "sports ball" class WILL miss frames. We run at
   `imgsz=1280` and linearly interpolate gaps. If it's still flaky on a clip, the
   arrows do **not** depend on the ball — they use the tracked *player* coordinates,
   so the playbook frames still work. Ball overlay is a bonus, not a dependency.
2. **No auto pass-detection in v1.** You mark the release frame and the reception
   frame (two numbers) plus passer/receiver IDs. The code draws the arrows. This is
   the robust path; automate later if v1 lands.
3. **2010/11 footage is lowish-res** — expect missed detections in crowded midfield.
   Through-ball moments happen in open space, where tracking is cleanest.
4. **Footage rights:** this is UEFA/broadcast footage. Keep excerpts short and
   transformative — your tactical voiceover over brief clips. Flagging, not lecturing.

## Target moments
- **UCL Final 2011** (Barça 3–1 Man Utd): Pedro's opener ~27' — Xavi through ball, right channel.
- **UCL Semi 2011 2nd leg** (Barça 1–1 Madrid): Pedro ~54' — Iniesta releases him in behind.
- **La Liga 5–0 2010** (Barça 5–0 Madrid): Villa's two goals ~55' & ~58' — balls in behind the line.
''')

# ---------------------------------------------------------------------------
# CELL 1 — Setup
# ---------------------------------------------------------------------------
md("## Cell 1 — Setup (installs + Drive mount + folders)")
code(r'''# Run once per Colab session. ~3 min. Use a GPU runtime:
#   Runtime > Change runtime type > Hardware accelerator = T4 GPU
import sys, subprocess

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("supervision>=0.21.0", "ultralytics>=8.2.0", "yt-dlp", "opencv-python-headless", "pandas")

import os, cv2, numpy as np, supervision as sv
print("supervision", sv.__version__, "| OpenCV", cv2.__version__)

# Confirm GPU
try:
    import torch
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    if not torch.cuda.is_available():
        print("⚠️  No GPU. Tracking will be SLOW. Switch runtime to T4 GPU and re-run.")
except Exception as e:
    print("torch check skipped:", e)

# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = "/content/drive/MyDrive/soccer_through_ball"
CLIPS_DIR   = os.path.join(PROJECT_DIR, "clips")
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs")
FRAMES_DIR  = os.path.join(PROJECT_DIR, "freezeframes")
for d in (PROJECT_DIR, CLIPS_DIR, OUT_DIR, FRAMES_DIR):
    os.makedirs(d, exist_ok=True)
print("✅ Output folders ready under", PROJECT_DIR)
''')

# ---------------------------------------------------------------------------
# CELL 2 — Download + trim with preview
# ---------------------------------------------------------------------------
md(r'''## Cell 2 — Download + trim (with frame preview so you confirm the window)

**Workflow per clip:**
1. `download_match("ucl_final")` → grabs the highlight reel to `/content`.
2. `preview(raw_path, seconds=[...])` → shows candidate frames. **Find the moment just before the pass.**
3. `trim(raw_path, start="MM:SS", dur=18, name="clip_pedro_final")` → saves the trimmed clip to Drive.

⚠️ Timestamps inside highlight reels vary by upload. Always preview before trimming — don't guess silently.
''')
code(r'''import subprocess, os, math
import cv2, numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Edit search queries here if the auto-pick grabs the wrong video.
MATCHES = {
    "ucl_final": {
        "query": "Barcelona Manchester United 2011 Champions League final highlights",
        "raw": "/content/raw_ucl_final.mp4",
        "note": "Pedro opener ~27' — Xavi through ball, right channel",
    },
    "clasico_5_0": {
        "query": "Barcelona Real Madrid 5-0 2010 highlights",
        "raw": "/content/raw_clasico_5_0.mp4",
        "note": "Villa's 2 goals ~55' & ~58' — balls in behind the line",
    },
    "ucl_semi": {
        "query": "Barcelona Real Madrid 2011 Champions League semifinal second leg highlights",
        "raw": "/content/raw_ucl_semi.mp4",
        "note": "Pedro ~54' — Iniesta releases him in behind",
    },
}

def download_match(key):
    m = MATCHES[key]
    print(f"⬇️  {key}: {m['note']}")
    if os.path.exists(m["raw"]):
        print("   already downloaded:", m["raw"]); return m["raw"]
    cmd = ["yt-dlp", "-f", "mp4[height<=720]/best[height<=720]",
           "-o", m["raw"], f"ytsearch1:{m['query']}"]
    subprocess.run(cmd, check=True)
    print("   saved:", m["raw"]); return m["raw"]

def _hms(t):  # seconds -> "MM:SS" / "HH:MM:SS"
    t = int(t); h, r = divmod(t, 3600); mn, s = divmod(r, 60)
    return f"{h:02d}:{mn:02d}:{s:02d}" if h else f"{mn:02d}:{s:02d}"

def video_len(path):
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release(); return n / fps

def preview(path, seconds=None, step=None):
    """Show candidate frames. Pass explicit `seconds=[...]` or a `step` (e.g. 10)
    to sample the whole reel every N seconds and locate the moment."""
    dur = video_len(path)
    if seconds is None:
        step = step or max(5, int(dur // 12))
        seconds = list(range(0, int(dur), step))
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = len(seconds); cols = 3; rows = math.ceil(n / cols)
    plt.figure(figsize=(15, 4 * rows))
    for i, sec in enumerate(seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, fr = cap.read()
        ax = plt.subplot(rows, cols, i + 1); ax.axis("off")
        if ok:
            ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{_hms(sec)}  ({sec}s)", fontsize=11)
    cap.release(); plt.tight_layout(); plt.show()
    print(f"Reel length: {_hms(dur)} ({dur:.0f}s). Pick a start a few sec BEFORE the pass.")

def trim(path, start, dur=18, name="clip"):
    """start: 'MM:SS' or seconds. Saves clip_<name>.mp4 to Drive CLIPS_DIR."""
    out = os.path.join(CLIPS_DIR, name if name.endswith(".mp4") else name + ".mp4")
    start = _hms(start) if isinstance(start, (int, float)) else start
    # Re-encode (not -c copy) so the clip starts on a clean keyframe — matters for frame-accurate IDs.
    cmd = ["ffmpeg", "-y", "-ss", start, "-i", path, "-t", str(dur),
           "-c:v", "libx264", "-preset", "veryfast", "-an", out]
    subprocess.run(cmd, check=True)
    print("✂️  saved trimmed clip:", out, "(", video_len(out), "s )")
    preview(out, step=2)   # quick look at the trimmed result
    return out

print("Helpers ready: download_match(key), preview(path, seconds=/step=), trim(path, start, dur, name)")
print("Keys:", list(MATCHES))
''')
md(r'''**Example — do these one at a time, confirming each preview:**
```python
raw = download_match("ucl_final")
preview(raw, step=10)            # scan the whole reel, find Pedro's goal
preview(raw, seconds=[85,88,90,92,94,96])   # zoom into the candidate window
clip = trim(raw, start="01:28", dur=18, name="clip_pedro_final")  # start ~3s before the pass
```
''')

# ---------------------------------------------------------------------------
# CELL 3 — Detect + track
# ---------------------------------------------------------------------------
md(r'''## Cell 3 — Detect + track (players + ball → annotated MP4 + coord cache)

Get this clean on **one** clip before batching. Produces:
- `outputs/<name>_tracked.mp4` — broadcast ellipses, tracker IDs, run traces, ball dot.
- `outputs/<name>_track.pkl` — per-frame player/ball coordinates (Cell 5 reads this for arrows).
''')
code(r'''import pickle, os, numpy as np, cv2, supervision as sv
from ultralytics import YOLO

PERSON, BALL = 0, 32
_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8x.pt")   # downloads ~130MB first time
    return _model

def _interpolate_ball(ball_by_frame, n_frames, max_gap=20):
    """ball_by_frame: {idx: (x,y)}. Linear-fill short gaps so the dot doesn't flicker."""
    idxs = sorted(ball_by_frame)
    if len(idxs) < 2:
        return dict(ball_by_frame)
    out = dict(ball_by_frame)
    for a, b in zip(idxs, idxs[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            (x0, y0), (x1, y1) = ball_by_frame[a], ball_by_frame[b]
            for k in range(1, gap):
                t = k / gap
                out[a + k] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return out

def track_clip(clip_path, imgsz=1280, conf=0.3, trace_len=90):
    """Detect+track a clip. Writes *_tracked.mp4 and *_track.pkl to OUT_DIR. Returns paths."""
    name = os.path.splitext(os.path.basename(clip_path))[0]
    out_mp4 = os.path.join(OUT_DIR, f"{name}_tracked.mp4")
    out_pkl = os.path.join(OUT_DIR, f"{name}_track.pkl")

    model = get_model()
    tracker = sv.ByteTrack()
    ellipse = sv.EllipseAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
    labeler = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK,
                                text_scale=0.5, text_thickness=1,
                                text_position=sv.Position.BOTTOM_CENTER)
    tracer  = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK,
                                trace_length=trace_len, thickness=2)

    players = {}   # frame_idx -> {tracker_id: (cx, cy)}
    ball_raw = {}  # frame_idx -> (cx, cy)

    def callback(frame, idx):
        res = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)

        # Ball: highest-confidence sports-ball box for this frame (before tracking).
        ball = det[det.class_id == BALL]
        if len(ball):
            bi = int(np.argmax(ball.confidence))
            bx1, by1, bx2, by2 = ball.xyxy[bi]
            ball_raw[idx] = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        # Players: track only the 'person' class for stable IDs.
        people = det[det.class_id == PERSON]
        people = tracker.update_with_detections(people)

        coords = {}
        if people.tracker_id is not None:
            for box, tid in zip(people.xyxy, people.tracker_id):
                x1, y1, x2, y2 = box
                coords[int(tid)] = ((x1 + x2) / 2, (y1 + y2) / 2)
        players[idx] = coords

        out = ellipse.annotate(frame.copy(), people)
        out = tracer.annotate(out, people)
        if people.tracker_id is not None:
            out = labeler.annotate(out, people, labels=[f"#{int(t)}" for t in people.tracker_id])
        if idx in ball_raw:
            bx, by = map(int, ball_raw[idx])
            cv2.circle(out, (bx, by), 6, (0, 255, 255), -1)
            cv2.circle(out, (bx, by), 7, (0, 0, 0), 1)
        return out

    print(f"🎬 tracking {name} (imgsz={imgsz}) ...")
    sv.process_video(source_path=clip_path, target_path=out_mp4, callback=callback)

    info = sv.VideoInfo.from_video_path(clip_path)
    ball = _interpolate_ball(ball_raw, info.total_frames)
    with open(out_pkl, "wb") as f:
        pickle.dump({"players": players, "ball": ball, "fps": info.fps,
                     "size": (info.width, info.height), "tracked_mp4": out_mp4,
                     "source_clip": clip_path}, f)

    hit = len(ball_raw); tot = info.total_frames
    print(f"✅ {out_mp4}")
    print(f"   ball detected on {hit}/{tot} frames ({100*hit/max(tot,1):.0f}%); "
          f"interpolated to {len(ball)}.")
    if hit / max(tot, 1) < 0.25:
        print("   ⚠️ Ball detection weak on this clip — that's expected on broadcast footage. "
              "Arrows use player coords, so playbook frames are unaffected.")
    return out_mp4, out_pkl

print("Ready: track_clip(clip_path).  e.g. track_clip(os.path.join(CLIPS_DIR,'clip_pedro_final.mp4'))")
''')

# ---------------------------------------------------------------------------
# CELL 4 — Read IDs (contact sheet)
# ---------------------------------------------------------------------------
md(r'''## Cell 4 — Read tracker IDs off the tracked video

Before drawing arrows you need the **passer** and **receiver** tracker IDs. This
shows tracked frames across the clip with the `#ID` labels visible. Scrub to the
moment of the pass, read the two numbers, and note the **release frame** and
**reception frame** indices.
''')
code(r'''import cv2, math, os, pickle
import matplotlib.pyplot as plt

def show_tracked(name_or_pkl, frames=None, step=None):
    """Contact sheet from the tracked MP4. `frames=[i,...]` exact indices, or `step` to sample."""
    pkl = name_or_pkl if name_or_pkl.endswith(".pkl") else os.path.join(OUT_DIR, name_or_pkl + "_track.pkl")
    data = pickle.load(open(pkl, "rb"))
    path = data["tracked_mp4"]
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None:
        step = step or max(1, total // 12)
        frames = list(range(0, total, step))
    cols = 3; rows = math.ceil(len(frames) / cols)
    plt.figure(figsize=(16, 4.2 * rows))
    for i, fi in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
        ax = plt.subplot(rows, cols, i + 1); ax.axis("off")
        if ok:
            ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"frame {fi}", fontsize=11)
    cap.release(); plt.tight_layout(); plt.show()
    print(f"{path}: {total} frames total. Note release & reception frame numbers + the two #IDs.")

def ids_at(name_or_pkl, frame_idx):
    """List tracker IDs present at a given frame (sanity check before drawing)."""
    pkl = name_or_pkl if name_or_pkl.endswith(".pkl") else os.path.join(OUT_DIR, name_or_pkl + "_track.pkl")
    data = pickle.load(open(pkl, "rb"))
    print(f"frame {frame_idx} IDs:", sorted(data["players"].get(frame_idx, {})))

print("Ready: show_tracked(name, frames=/step=) , ids_at(name, frame_idx)")
''')

# ---------------------------------------------------------------------------
# CELL 5 — Arrows / freeze-frames
# ---------------------------------------------------------------------------
md(r'''## Cell 5 — Playbook freeze-frames (pass arrow + run arrow + space zone)

Give it: `name`, `release_frame`, `reception_frame`, `passer_id`, `receiver_id`.
It draws onto the tracked frame (ellipses/IDs/traces already burned in):
- **Pass lane** — arrow from passer (at release) → receiver (at reception).
- **Run into space** — arrow along the receiver's path from release → reception.
- **Space zone** — translucent shaded area the run attacked.

All coordinates come from the tracked player cache, so the ball is not required.
''')
code(r'''import cv2, os, pickle, numpy as np

# BGR colors
C_PASS = (0, 215, 255)    # amber  — pass lane
C_RUN  = (60, 220, 60)    # green  — receiver's run
C_ZONE = (0, 140, 255)    # orange — space zone fill

def _path_between(players, a, b, tid):
    """Receiver center for every frame in [a,b] where the ID exists."""
    pts = []
    for fi in range(min(a, b), max(a, b) + 1):
        c = players.get(fi, {}).get(tid)
        if c:
            pts.append((int(c[0]), int(c[1])))
    return pts

def _zone_poly(p_run_start, p_run_end, width=70):
    """Channel quad around the run vector, flared toward the destination."""
    x0, y0 = p_run_start; x1, y1 = p_run_end
    dx, dy = x1 - x0, y1 - y0
    L = max((dx*dx + dy*dy) ** 0.5, 1e-6)
    nx, ny = -dy / L, dx / L          # unit normal
    w0, w1 = width * 0.5, width * 1.4 # narrow at start, flared at the space
    return np.array([
        [x0 + nx*w0, y0 + ny*w0],
        [x1 + nx*w1, y1 + ny*w1],
        [x1 - nx*w1, y1 - ny*w1],
        [x0 - nx*w0, y0 - ny*w0],
    ], dtype=np.int32)

def playbook_frame(name_or_pkl, release_frame, reception_frame, passer_id, receiver_id,
                   use_frame="reception", zone_width=70, tag=None, show=True):
    """Render one playbook PNG. use_frame: which frame to draw on ('reception' or 'release')."""
    pkl = name_or_pkl if name_or_pkl.endswith(".pkl") else os.path.join(OUT_DIR, name_or_pkl + "_track.pkl")
    data = pickle.load(open(pkl, "rb"))
    players, tracked = data["players"], data["tracked_mp4"]

    draw_idx = reception_frame if use_frame == "reception" else release_frame
    cap = cv2.VideoCapture(tracked)
    cap.set(cv2.CAP_PROP_POS_FRAMES, draw_idx)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {draw_idx} from {tracked}")

    passer_at_release = players.get(release_frame, {}).get(passer_id)
    recv_at_release   = players.get(release_frame, {}).get(receiver_id)
    recv_at_recept    = players.get(reception_frame, {}).get(receiver_id)
    missing = [n for n, v in [("passer@release", passer_at_release),
                              ("receiver@release", recv_at_release),
                              ("receiver@reception", recv_at_recept)] if v is None]
    if missing:
        print("⚠️ missing coords:", missing,
              "— check IDs with ids_at(), or nudge the frame numbers by ±1–2.")

    overlay = frame.copy()

    # Space zone (under the arrows) along the run.
    if recv_at_release and recv_at_recept:
        poly = _zone_poly(tuple(map(int, recv_at_release)), tuple(map(int, recv_at_recept)), zone_width)
        cv2.fillPoly(overlay, [poly], C_ZONE)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        cv2.polylines(frame, [poly], True, C_ZONE, 2)

    # Run arrow — follow the traced path, head at reception.
    if recv_at_release and recv_at_recept:
        path = _path_between(players, release_frame, reception_frame, receiver_id)
        if len(path) >= 2:
            for a, b in zip(path, path[1:]):
                cv2.line(frame, a, b, C_RUN, 3)
            cv2.arrowedLine(frame, path[-2], path[-1], C_RUN, 4, tipLength=0.5)
        else:
            cv2.arrowedLine(frame, tuple(map(int, recv_at_release)),
                            tuple(map(int, recv_at_recept)), C_RUN, 4, tipLength=0.3)

    # Pass arrow — passer(release) -> receiver(reception).
    if passer_at_release and recv_at_recept:
        cv2.arrowedLine(frame, tuple(map(int, passer_at_release)),
                        tuple(map(int, recv_at_recept)), C_PASS, 4, tipLength=0.18)

    # Legend
    for i, (txt, col) in enumerate([("PASS LANE", C_PASS), ("RUN INTO SPACE", C_RUN), ("SPACE", C_ZONE)]):
        y = 28 + i * 26
        cv2.rectangle(frame, (12, y - 14), (32, y + 4), col, -1)
        cv2.putText(frame, txt, (40, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    base = os.path.splitext(os.path.basename(pkl))[0].replace("_track", "")
    tag = tag or f"{release_frame}-{reception_frame}"
    out_png = os.path.join(FRAMES_DIR, f"{base}_playbook_{tag}.png")
    cv2.imwrite(out_png, frame)
    print("🖼️  saved", out_png)
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 8)); plt.axis("off")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.show()
    return out_png

print("Ready: playbook_frame(name, release_frame, reception_frame, passer_id, receiver_id)")
''')
md(r'''**Example:**
```python
# after reading IDs in Cell 4 — Xavi #7 releases Pedro #11:
playbook_frame("clip_pedro_final",
               release_frame=72, reception_frame=96,
               passer_id=7, receiver_id=11,
               use_frame="reception", zone_width=80, tag="pedro_opener")
```
Tweak `zone_width`, swap `use_frame="release"`, or nudge the frame numbers ±2 to get the cleanest still.
''')

# ---------------------------------------------------------------------------
# CELL 6 — Batch
# ---------------------------------------------------------------------------
md(r'''## Cell 6 — Batch the remaining clips

Once clip #1 looks clean, list every trimmed clip in `clips/` and track them all.
Re-run is cheap to skip already-done ones.
''')
code(r'''import glob, os

def batch_track(overwrite=False, **kw):
    clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))
    if not clips:
        print("No clips in", CLIPS_DIR, "— run Cell 2 first."); return
    print("Found clips:", [os.path.basename(c) for c in clips])
    for c in clips:
        name = os.path.splitext(os.path.basename(c))[0]
        done = os.path.join(OUT_DIR, f"{name}_tracked.mp4")
        if os.path.exists(done) and not overwrite:
            print("⏭️  skip (already tracked):", name); continue
        track_clip(c, **kw)
    print("✅ batch done.")

# batch_track()                 # track everything not yet done
# batch_track(overwrite=True)   # force re-track all
print("Ready: batch_track(overwrite=False)")
''')

# ---------------------------------------------------------------------------
# CELL 7 — Export / summary
# ---------------------------------------------------------------------------
md(r'''## Cell 7 — Export summary (what's in Drive, ready for voiceover)

Lists the final tracked MP4s and playbook PNGs with sizes. Download from the Drive
folder and drop into CapCut; record your tactical voiceover over the short excerpts.
''')
code(r'''import glob, os

def human(n):
    for u in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def summary():
    print("📁", PROJECT_DIR, "\n")
    for label, pat in [("Trimmed clips", os.path.join(CLIPS_DIR, "*.mp4")),
                       ("Tracked videos", os.path.join(OUT_DIR, "*_tracked.mp4")),
                       ("Coord caches", os.path.join(OUT_DIR, "*_track.pkl")),
                       ("Playbook PNGs", os.path.join(FRAMES_DIR, "*_playbook_*.png"))]:
        files = sorted(glob.glob(pat))
        print(f"— {label} ({len(files)}):")
        for f in files:
            print(f"    {os.path.basename(f):45s} {human(os.path.getsize(f))}")
        if not files: print("    (none yet)")
        print()
    print("Done. Download from Drive →", PROJECT_DIR, "→ CapCut → voiceover. Keep excerpts short.")

summary()
''')

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Normalize source: nbformat wants a list of lines (each ending in \n except last).
for c in nb["cells"]:
    s = c["source"]
    if isinstance(s, str):
        lines = s.splitlines(keepends=True)
        c["source"] = lines

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", NB_PATH, "with", len(cells), "cells")
