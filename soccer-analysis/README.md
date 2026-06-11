# Soccer Through-Ball Analysis (v1)

Computer-vision pipeline that turns broadcast footage of Barcelona's 2010–11
through-ball goals into coach-style playbook breakdowns: tracked videos
(player ellipses + IDs + run trails + ball marker) and freeze-frame PNGs with
pass arrow, run arrow, and a shaded zone over the attacked space.

**Open the notebook in Colab (needs a free T4 GPU runtime):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassanmuh467-design/TradingAgents/blob/claude/soccer-through-ball-analysis-nu6hep/soccer-analysis/through_ball_playbook.ipynb)

## How it works

| Cell | What it does | Your input |
|---|---|---|
| 1 | installs, mounts Drive, helpers | run it every session |
| 2 | downloads the 3 highlight videos | swap in a direct URL if a search hit is wrong |
| 3 | frame-grid scrubber to find each moment | a time window |
| 4 | ffmpeg-trims the 4 clips | start time + length per clip |
| 5–6 | YOLOv8x detect + ByteTrack, tracked MP4 | one clip name |
| 7 | optional manual ball fix | only if ball rate < 30% |
| 8 | playbook freeze-frames | 4 numbers per clip (2 frames, 2 tracker IDs) |
| 9–10 | batch the rest, list Drive outputs | — |

All outputs are written immediately to `MyDrive/soccer_analysis/`, so nothing is
lost if the Colab session dies. Full instructions are in the notebook's top cell.

## Stack

`supervision==0.28.0`, `trackers==2.4.0` (`ByteTrackTracker` — supervision's own
`sv.ByteTrack` is deprecated and removed in 0.30), `ultralytics` YOLOv8x
(COCO class 0 = person, 32 = sports ball), `yt-dlp`, ffmpeg, OpenCV.

v1 scope is through-ball execution only; pass key-frames are user-marked by
design. Phase 2 (build-up patterns, midfield rotations) comes later.
