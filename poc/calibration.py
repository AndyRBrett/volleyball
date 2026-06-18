#!/usr/bin/env python3
"""Court calibration: map image pixels to real court coordinates (ground plane).

Given the 4 court corners clicked in an image, we build a homography that maps
image points to real-world court metres. Used to:

  * filter players to those actually *on the court* (drops background people /
    other courts) — point-in-court test on each player's foot position;
  * report player positions, distances, and a top-down heatmap in real metres.

Accurate for points ON THE GROUND (player feet). NOT valid for an airborne ball
(parallax) — ball metrics stay in pixels. Also assumes the camera is fixed for
the duration of the clip; if it moves, a single homography drifts and results are
only approximate (we flag this).

Corner order (matches the UI prompt): near-left, near-right, far-right, far-left.
Court size presets (length × width, metres):
  beach  16 × 8     indoor 18 × 9
"""

from __future__ import annotations

import math

COURTS = {"beach": (16.0, 8.0), "indoor": (18.0, 9.0)}
IN_COURT_MARGIN_M = 1.0   # tolerance outside the lines (warm-up space, line judges)
MIN_FRAMES_IN_COURT = 3   # ignore fleeting in-court blips when counting players
MAX_PLAYER_SPEED_MS = 11.0  # ~40 km/h; faster than a sprint -> detection jitter
HEATMAP_COLS = 24
HEATMAP_ROWS = 12
MS_TO_KMH = 3.6
MS_TO_MPH = 2.23694


def homography(corners_img: list[list[float]], length: float, width: float):
    """4 image corners -> 3x3 matrix mapping image px to court metres.

    corners_img order: near-left, near-right, far-right, far-left, matched to
    court points (0,0), (L,0), (L,W), (0,W).
    """
    import numpy as np

    src = np.array(corners_img, dtype=np.float64)
    dst = np.array([[0, 0], [length, 0], [length, width], [0, width]],
                   dtype=np.float64)
    # Solve the 8 DLT equations for the homography (no OpenCV dependency here).
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def to_court(H, x: float, y: float) -> tuple[float, float]:
    import numpy as np
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def in_court(cx: float, cy: float, length: float, width: float) -> bool:
    return (-IN_COURT_MARGIN_M <= cx <= length + IN_COURT_MARGIN_M and
            -IN_COURT_MARGIN_M <= cy <= width + IN_COURT_MARGIN_M)


def calibrate_players(events, corners, court, fps, camera_moves):
    """Recompute player metrics in real court metres, filtered to the court."""
    length, width = COURTS.get(court, COURTS["beach"])
    H = homography(corners, length, width)

    grid = [[0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)]
    last: dict[int, tuple[float, float, float]] = {}   # cx, cy, time_s
    dist: dict[int, float] = {}
    top_ms: dict[int, float] = {}
    seen: dict[int, int] = {}

    for e in events:
        t = e.get("time_s", 0.0)
        for pl in e.get("players", []):
            x1, y1, x2, y2 = pl["bbox"]
            fx, fy = (x1 + x2) / 2.0, y2          # foot = bottom-center
            cx, cy = to_court(H, fx, fy)          # -> court metres
            if not in_court(cx, cy, length, width):
                continue
            col = min(HEATMAP_COLS - 1, max(0, int(cx / length * HEATMAP_COLS)))
            row = min(HEATMAP_ROWS - 1, max(0, int(cy / width * HEATMAP_ROWS)))
            grid[row][col] += 1
            tid = pl.get("track_id")
            if tid is None:
                continue
            seen[tid] = seen.get(tid, 0) + 1
            if tid in last:
                step = math.hypot(cx - last[tid][0], cy - last[tid][1])
                dist[tid] = dist.get(tid, 0.0) + step
                dt = t - last[tid][2]
                if dt > 0:
                    v = step / dt
                    if v <= MAX_PLAYER_SPEED_MS:   # reject jitter spikes
                        top_ms[tid] = max(top_ms.get(tid, 0.0), v)
            last[tid] = (cx, cy, t)

    per_track = [{"track_id": tid, "frames_in_court": seen[tid],
                  "distance_m": round(dist.get(tid, 0.0), 2),
                  "top_speed_kmh": round(top_ms.get(tid, 0.0) * MS_TO_KMH, 1),
                  "top_speed_mph": round(top_ms.get(tid, 0.0) * MS_TO_MPH, 1)}
                 for tid in seen if seen[tid] >= MIN_FRAMES_IN_COURT]
    per_track.sort(key=lambda p: p["distance_m"], reverse=True)

    return {
        "calibration": {
            "court": court, "court_m": [length, width],
            "camera_moves": bool(camera_moves),
            "approximate": bool(camera_moves),
            "corners_img": corners,
        },
        "players_court": {
            "track_count_in_court": len(per_track),
            "per_track": per_track,
            "heatmap": {"cols": HEATMAP_COLS, "rows": HEATMAP_ROWS,
                        "court_m": [length, width], "grid": grid},
        },
    }
