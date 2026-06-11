# ⚽ Through-Ball Analysis — Barcelona 2010/11

Computer-vision pipeline that turns short broadcast clips of **through-ball
sequences** into coach-style playbook breakdowns using Roboflow's
[Supervision](https://github.com/roboflow/supervision) + YOLOv8.

**Output per clip:** a tracked MP4 (broadcast ellipses, tracker IDs, run traces,
ball dot) plus 2–3 annotated freeze-frame PNGs with a **pass-lane arrow**, a
**run-into-space arrow**, and a shaded **space zone** — ready for you to record a
tactical voiceover over in CapCut.

## Quick start

1. Open **`soccer_through_ball_analysis.ipynb`** in
   [Google Colab](https://colab.research.google.com/) (phone or laptop).
2. Set a GPU runtime: *Runtime → Change runtime type → T4 GPU*.
3. Run the cells top to bottom. Each is self-contained and saves straight to
   Google Drive, so a dropped session loses nothing.

| Cell | Does | You do |
|---|---|---|
| 1 Setup | installs + Drive mount + folders | run it (~3 min) |
| 2 Download + trim | yt-dlp search → ffmpeg trim, **with frame preview** | confirm the window before trimming |
| 3 Detect + track | players + ball, ByteTrack IDs → MP4 + coord cache | run on ONE clip first |
| 4 Read IDs | contact sheet of tracked frames | note passer/receiver `#IDs` + frame numbers |
| 5 Arrows | playbook freeze-frame PNGs | give 2 frame numbers + 2 IDs |
| 6 Batch | track the remaining clips | run after clip #1 looks clean |
| 7 Export | lists everything in Drive | download → CapCut → voiceover |

## Outputs (Google Drive)

```
MyDrive/soccer_through_ball/
├── clips/          trimmed source clips        (clip_*.mp4)
├── outputs/        tracked videos + caches     (*_tracked.mp4, *_track.pkl)
└── freezeframes/   playbook stills             (*_playbook_*.png)
```

## Target moments

| Match | Date | Moment |
|---|---|---|
| Barça 3–1 Man Utd, UCL Final (Wembley) | 28 May 2011 | Pedro ~27' — Xavi through ball, right channel |
| Barça 1–1 Real Madrid, UCL Semi 2nd leg | 3 May 2011 | Pedro ~54' — Iniesta releases him in behind |
| Barça 5–0 Real Madrid, La Liga | 29 Nov 2010 | Villa ~55' & ~58' — balls in behind the line |

YouTube URLs aren't pinned; Cell 2 resolves them via `yt-dlp` search and
**previews candidate frames so you confirm the right window** before trimming.

## Honest caveats (read before relying on it)

- **Ball detection on broadcast footage is unreliable** — tiny, fast,
  motion-blurred. We run at `imgsz=1280` and interpolate gaps, but it *will* miss
  frames. The arrows use tracked **player** coordinates, not the ball, so the
  playbook frames are unaffected if the ball flickers.
- **No automatic pass detection in v1.** You mark the release frame, the reception
  frame, and the two tracker IDs; the code draws the arrows. Robust now, automate
  later.
- **2010/11 footage is lowish-res** — expect missed detections in crowded
  midfield. Through-ball moments live in open space, where tracking is cleanest.
- **Footage rights:** UEFA/broadcast material. Keep excerpts short and
  transformative (your analysis + voiceover over brief clips).

## Scope

**v1 (this build):** through-ball execution only.
**Phase 2 (later):** build-up / possession patterns and midfield rotations.

## Editing the notebook

Cell sources live in **`build_notebook.py`** as plain strings. Edit there and
regenerate:

```bash
python3 build_notebook.py
```

## Stack

`supervision` · `ultralytics` (YOLOv8x; class 0 = person, 32 = sports ball) ·
`sv.ByteTrack` · `EllipseAnnotator` / `LabelAnnotator` / `TraceAnnotator` ·
`yt-dlp` · `opencv-python` · Google Colab (T4).

Reference patterns: [roboflow/sports](https://github.com/roboflow/sports).
