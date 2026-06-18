# TODO / parking lot

Deferred ideas to revisit. Active build order lives in the README "Status"
section; this file is for things we've consciously postponed.

## Cut the Roboflow API dependency (run ball detection locally)

**Status:** deferred (chose to keep the hosted API while building metrics/coaching).

Today, ball detection calls the Roboflow-hosted model **once per processed
frame**. That's fine for the handful of clips we process while building features,
but for routine use it means: monthly free-tier quota, internet required,
network-bound speed (~3 fps), and frames leaving the machine.

**Goal:** run the ball model locally (a volleyball-trained YOLOv8 `.pt`), exactly
like the local `yolov8n.pt` player model — fully offline, no API calls, faster,
private.

**Work:**
- Add a `--ball-weights path/to/model.pt` option to `poc/pipeline.py` that runs
  the ball model via local `ultralytics` instead of Roboflow (use the model's own
  `ball`/`volleyball` class index). Keep `--rf-model` as the fallback.
- Obtain a local `.pt`, whichever is easier:
  - download a YOLOv8 weights file if a Roboflow project / HF repo offers one, or
  - export the public `volleyball_detection` dataset from Roboflow and train
    YOLOv8-nano in a free Colab GPU (~30 min, one-time) → `best.pt`.
- Once local, the whole pipeline runs offline; the webapp no longer needs
  `ROBOFLOW_API_KEY`.

## Moving-camera court calibration (per-frame)

**Status:** deferred. A single 4-corner calibration is accurate only while the
camera is fixed for the clip. When the camera pans/zooms during a clip (common
for handheld phone footage), the homography drifts and court-filtering + real
units become approximate — the UI flags this when "camera moves" is ticked.

**Real fix:** track the court across frames so the homography updates per frame —
e.g. detect court lines/corners each frame, or track features / stabilize the
video and propagate the initial homography. This is a meaningful CV effort;
until then, fixed-camera footage gives accurate metrics and moving-camera footage
is labelled approximate.

## Real-world ball speed (3D)

**Status:** deferred. The ground-plane homography is correct for player feet but
not for an airborne ball (parallax), so ball speed stays in pixels. Real ball
speed needs 3D reconstruction (multi-view, or a physics/arc model with known
court geometry). Player metres/speed are unaffected.

## Housekeeping

- **Regenerate the Roboflow API key** that was shared during setup (it was pasted
  into a chat). Roboflow → Settings → API Keys → regenerate.
- **Player ID churn (confirmed).** On a longer/busier clip the tracker reported
  ~47 distinct IDs for a handful of real players — ByteTrack spawns new IDs when
  players cluster at the net, cross, or briefly leave frame. The "players tracked"
  metric is inflated as a result. Fixes to try: tune ByteTrack
  (`track_buffer`, match thresholds), use a stronger detector (yolov8s/m), or add
  re-identification. Until then, treat track_count as an upper bound and consider
  reporting only tracks seen in >= N frames.
