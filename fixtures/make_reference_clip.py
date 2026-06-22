#!/usr/bin/env python3
"""Generate the bundled reference clip used by the pipeline self-test.

Why a synthetic clip?
---------------------
The self-test needs a *real* set of pixel frames the detector can run on, small
enough to commit and decodable with only the standard library (no ffmpeg, no
opencv, no GPU). So instead of a multi-megabyte MP4 we render a short clip as a
sequence of grayscale Netpbm (P5/PGM) frames -- a dark court with one bright
disc (the "ball") arcing across it -- and gzip the lot. ``detect.py`` then runs
an honest brightest-blob centroid detector over those pixels to recover the
ball track, exercising the full pipeline end-to-end.

The clip is deliberately structured into three rallies separated by gaps (all
-dark frames where the ball is out of play), so segmentation, tagging, speed,
and the contact-zone heatmap all have something real to chew on.

Outputs (committed as fixtures, regenerate with ``python make_reference_clip.py``):
  reference_clip.pgm.gz       -- gzipped P5/PGM frame sequence (the "clip").
  reference_clip.events.json  -- ground-truth coaching events bundled with it.

This generator is provenance/regeneration tooling; the self-test reads the
committed fixtures, not this script.
"""
import gzip
import json
import os

WIDTH = 80
HEIGHT = 45
FPS = 10
BALL_RADIUS = 2
BALL_VALUE = 255

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_PATH = os.path.join(HERE, "reference_clip.pgm.gz")
EVENTS_PATH = os.path.join(HERE, "reference_clip.events.json")

# Each rally is a run of frames carrying the ball; gaps between them are all
# -dark frames (ball out of play) longer than the segmentation gap threshold.
# (start_frame, end_frame_inclusive, x0, x1, y_low, y_peak) -- a horizontal
# sweep x0->x1 with a parabolic vertical arc dipping to y_peak and back.
_RALLIES = [
    (0, 17, 8, 70, 30, 10),    # rally A: left -> right
    (41, 63, 70, 8, 28, 8),    # rally B: right -> left (after a ~2.3s gap)
    (87, 103, 10, 60, 32, 12),  # rally C: left -> right (after a ~2.3s gap)
]
TOTAL_FRAMES = _RALLIES[-1][1] + 1

# Ground-truth coaching events, each timed to land inside a rally window so the
# tagger and contact-zone heatmap have real contacts to bin. Times are seconds.
_EVENTS = [
    {"t": 0.1, "type": "serve", "player": 4},
    {"t": 0.8, "type": "set", "player": 6},
    {"t": 1.4, "type": "attack", "player": 9},
    {"t": 4.3, "type": "dig", "player": 2},
    {"t": 5.2, "type": "set", "player": 6},
    {"t": 6.0, "type": "attack", "player": 11},
    {"t": 6.2, "type": "block", "player": 5},
    {"t": 8.9, "type": "reception", "player": 3},
    {"t": 9.8, "type": "attack", "player": 9},
]


def _ball_center(frame_idx):
    """Return the (cx, cy) ball center for a frame, or None when out of play."""
    for start, end, x0, x1, y_low, y_peak in _RALLIES:
        if start <= frame_idx <= end:
            span = max(1, end - start)
            p = (frame_idx - start) / span  # 0..1 across the rally
            cx = x0 + (x1 - x0) * p
            # Parabolic arc: dips from y_low down to y_peak at mid-rally and back.
            cy = y_low + (y_peak - y_low) * (1 - (2 * p - 1) ** 2)
            return cx, cy
    return None


def _render_frame(frame_idx):
    """Render one frame as a flat bytes buffer (row-major, 1 byte/pixel)."""
    buf = bytearray(WIDTH * HEIGHT)  # all dark
    center = _ball_center(frame_idx)
    if center is None:
        return bytes(buf)
    cx, cy = center
    icx, icy = int(round(cx)), int(round(cy))
    r2 = BALL_RADIUS * BALL_RADIUS
    for dy in range(-BALL_RADIUS, BALL_RADIUS + 1):
        py = icy + dy
        if not (0 <= py < HEIGHT):
            continue
        for dx in range(-BALL_RADIUS, BALL_RADIUS + 1):
            px = icx + dx
            if not (0 <= px < WIDTH):
                continue
            if dx * dx + dy * dy <= r2:
                buf[py * WIDTH + px] = BALL_VALUE
    return bytes(buf)


def build_clip_bytes():
    """Build the full gzipped P5/PGM frame-sequence container as bytes."""
    out = bytearray()
    header = b"P5\n%d %d\n255\n" % (WIDTH, HEIGHT)
    for i in range(TOTAL_FRAMES):
        out += header
        out += _render_frame(i)
    return gzip.compress(bytes(out), mtime=0)


def write_fixtures():
    with open(CLIP_PATH, "wb") as fh:
        fh.write(build_clip_bytes())
    events_doc = {"fps": FPS, "source": "fixtures/reference_clip.pgm.gz", "events": _EVENTS}
    with open(EVENTS_PATH, "w") as fh:
        json.dump(events_doc, fh, indent=2)
        fh.write("\n")
    return CLIP_PATH, EVENTS_PATH


def main():
    clip, events = write_fixtures()
    size = os.path.getsize(clip)
    print(f"Wrote {clip} ({TOTAL_FRAMES} frames, {size} bytes) and {events}")


if __name__ == "__main__":
    main()
