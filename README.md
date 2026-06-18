# volleyball

Video recognition and analysis for volleyball footage: detect the ball, track
players, compute metrics, and turn the results into coaching feedback.

This is an **offline batch tool**, not a real-time system — you record a clip,
the pipeline processes it once (slow on an old laptop, but you walk away), and
the UI plays back the *already-processed* result. That framing is deliberate: it
keeps the project runnable on modest hardware (e.g. a 2017 Intel MacBook, CPU
only).

## Architecture (target)

```
1. CAPTURE     phone records the court -> upload a video
2. PIPELINE    YOLOv8 ball + player detection, ByteTrack tracking,
               court homography -> structured JSON per clip
3. METRICS     rally length, court zones/heatmaps, ball speed, contact points
4. COACHING    feed structured stats to Claude (claude-opus-4-8) -> analysis
5. WEB UI      upload, annotated playback, coaching panel, session history
```

Build order follows risk: **validate ball detection first** (`poc/`), then the
metrics layer, then the coaching call, then the UI.

## Status

- [x] **PoC: ball + player detection** (`poc/detect_ball.py`) — pipeline runs
      end-to-end; player detection + ByteTrack tracking validated on real footage.
- [x] **Ball detection validated** — the stock COCO model can't see a volleyball
      (0%), so the ball comes from a Roboflow-hosted volleyball model
      (`volleyball_detection/2`, ~97% mAP). On real game footage it detects the
      ball in **~100% of frames where the ball is in view** (the only misses are
      when the ball is genuinely off-screen). This was the project's biggest risk
      and it's resolved.
- [x] **Combined pipeline** (`poc/pipeline.py`) — local player tracking + Roboflow
      ball detection in one pass; player IDs stay stable, ball marker tracks
      cleanly. Emits an annotated video + per-frame events JSON.
- [x] **Metrics layer** (`poc/metrics.py`) — ball-path interpolation, rally
      segmentation, ball speed, per-player distance, position heatmap (pixels).
- [x] **Web UI shell** (`webapp/`) — upload → background processing → annotated
      playback with ball-path overlay + metrics panels.
- [x] **Court calibration** (`poc/calibration.py`) — click 4 corners → filter
      players to the court (drops background people) + real-world units (metres).
      Accurate for a fixed camera; **approximate and flagged** when the camera
      moves during the clip (per-frame court tracking is a TODO).
- [ ] Claude coaching call (`claude-opus-4-8`) — written readout from the metrics
- [ ] Real-world ball speed (needs 3D; ball is airborne — ground homography only
      handles player feet). See TODO.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # macOS: pulls the correct CPU/Metal torch automatically

# Combined pipeline: players (local) + ball (Roboflow). Set your key first.
export ROBOFLOW_API_KEY=your_roboflow_key
python3 poc/pipeline.py -i your_clip.mp4 --rf-model volleyball_detection/2 --stride 5
```

Outputs `<clip>_pipeline.mp4` (overlay) and `<clip>_pipeline.json` (per-frame
events). Players-only, no key: add `--no-ball`. See [`poc/README.md`](poc/README.md)
for the detection-only PoC and all flags.

## How detection works (validated)

- **Players** — local YOLOv8 (`ultralytics`) `person` class + ByteTrack for stable
  IDs. Runs fully on-device; ~3–8 fps on a 2017 Intel MacBook.
- **Ball** — the stock COCO model has no volleyball, so the ball comes from a
  Roboflow-hosted volleyball model over HTTP (one call per processed frame; the
  network is the bottleneck). Detects the ball in essentially every in-view frame.
  Swapping in a local volleyball `.pt` later would remove the per-frame API call.

## Notes on models

Coaching feedback (step 4) uses Anthropic's current top Opus model,
**`claude-opus-4-8`** (1M-token context at standard pricing — enough to feed a
whole match's structured events in one call).
