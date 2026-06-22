#!/usr/bin/env python3
"""Ball detection: turn raw clip frames into tracking data (the CV front-end).

This is the stage that was missing -- the reason the pipeline had "processed 0
frames ever". Everything downstream (highlights.py, coaching.py) consumes
*tracking* data; nothing produced it from actual pixels. This module closes that
gap with a deliberately small, dependency-free detector so the pipeline can be
proven end-to-end in CI without ffmpeg, opencv, torch, or a GPU.

It reads a clip stored as a gzipped sequence of Netpbm P5/PGM frames (see
fixtures/make_reference_clip.py), and for each frame recovers the ball position
as the centroid of the brightest blob. Frames with no bright blob (ball out of
play) yield ``ball: null`` -- exactly the signal highlights.segment_rallies uses
to split play into rallies.

The output schema matches the tracking JSON the rest of the pipeline already
expects (see highlights.py), with ``width``/``height`` added so the coaching
heatmap knows the court dimensions.
"""
import gzip
import json
import os

DEFAULT_THRESHOLD = 200   # pixel value at/above which a pixel is "ball-bright"
DEFAULT_MIN_PIXELS = 3     # fewer bright pixels than this -> treat as no ball


def _open_maybe_gzip(path):
    """Open ``path`` for binary reading, transparently decompressing .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_pgm_frames(path):
    """Load a (possibly gzipped) P5/PGM frame sequence.

    Returns ``(width, height, frames)`` where ``frames`` is a list of ``bytes``
    buffers, one per frame, each ``width * height`` bytes (row-major, 1 byte per
    pixel). All frames in a clip are required to share dimensions.
    """
    with _open_maybe_gzip(path) as fh:
        data = fh.read()

    frames = []
    width = height = None
    pos = 0
    n = len(data)
    while pos < n:
        magic, pos = _read_token(data, pos)
        if magic != b"P5":
            raise ValueError(f"unsupported frame magic {magic!r} (expected P5)")
        w_tok, pos = _read_token(data, pos)
        h_tok, pos = _read_token(data, pos)
        maxval_tok, pos = _read_token(data, pos)
        w, h = int(w_tok), int(h_tok)
        # Exactly one whitespace byte separates the header from the raster.
        pos += 1
        size = w * h
        raster = data[pos:pos + size]
        if len(raster) != size:
            raise ValueError("truncated PGM raster")
        pos += size
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError("inconsistent frame dimensions in clip")
        frames.append(raster)
    if width is None:
        raise ValueError("no frames found in clip")
    return width, height, frames


def _read_token(data, pos):
    """Read one whitespace-delimited token from ``data`` starting at ``pos``."""
    n = len(data)
    while pos < n and data[pos] in b" \t\r\n":
        pos += 1
    start = pos
    while pos < n and data[pos] not in b" \t\r\n":
        pos += 1
    return data[start:pos], pos


def detect_ball(raster, width, height, threshold=DEFAULT_THRESHOLD, min_pixels=DEFAULT_MIN_PIXELS):
    """Return the ``[x, y]`` centroid of the brightest blob, or None.

    Pixels at/above ``threshold`` are treated as ball-bright; their centroid is
    the detected ball position. When fewer than ``min_pixels`` qualify the ball
    is considered out of frame and None is returned.
    """
    count = 0
    sum_x = 0
    sum_y = 0
    for idx, value in enumerate(raster):
        if value >= threshold:
            count += 1
            sum_x += idx % width
            sum_y += idx // width
    if count < min_pixels:
        return None
    return [round(sum_x / count, 2), round(sum_y / count, 2)]


def detect_frames(width, height, frames, threshold=DEFAULT_THRESHOLD, min_pixels=DEFAULT_MIN_PIXELS, fps=10.0):
    """Run the detector over every frame, returning per-frame tracking records."""
    records = []
    for i, raster in enumerate(frames):
        ball = detect_ball(raster, width, height, threshold=threshold, min_pixels=min_pixels)
        records.append({"frame": i, "t": round(i / fps, 4), "ball": ball})
    return records


def run_detection(
    clip_path,
    fps=10.0,
    events=None,
    source=None,
    threshold=DEFAULT_THRESHOLD,
    min_pixels=DEFAULT_MIN_PIXELS,
):
    """Detect the ball track in ``clip_path`` and return a tracking dict.

    The returned dict matches the tracking schema highlights.py consumes:
    ``fps``, ``source``, ``width``, ``height``, ``frames`` (per-frame ball
    position or null) and ``events`` (passed through, since event detection is
    out of scope for this geometric detector -- the reference clip bundles
    ground-truth events alongside the pixels).
    """
    width, height, frames = load_pgm_frames(clip_path)
    records = detect_frames(width, height, frames, threshold=threshold, min_pixels=min_pixels, fps=fps)
    detected = sum(1 for r in records if r["ball"] is not None)
    return {
        "fps": fps,
        "source": source if source is not None else clip_path,
        "width": width,
        "height": height,
        "frame_count": len(records),
        "detected_frames": detected,
        "frames": records,
        "events": list(events or []),
    }


def _load_events_sidecar(path):
    """Load a bundled events sidecar, returning (events, fps_or_None, source)."""
    with open(path) as fh:
        doc = json.load(fh)
    return doc.get("events", []), doc.get("fps"), doc.get("source")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect ball track from a clip and emit tracking JSON.")
    parser.add_argument("clip", help="Path to the clip (gzipped P5/PGM frame sequence)")
    parser.add_argument("--events", help="Optional bundled events sidecar JSON")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", help="Write tracking JSON here (default: stdout)")
    args = parser.parse_args()

    events = []
    fps = args.fps
    source = None
    if args.events:
        events, sidecar_fps, source = _load_events_sidecar(args.events)
        if sidecar_fps:
            fps = float(sidecar_fps)

    tracking = run_detection(args.clip, fps=fps, events=events, source=source)
    text = json.dumps(tracking, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
        print(f"Wrote {args.output}: {tracking['detected_frames']}/{tracking['frame_count']} frames with ball")
    else:
        print(text)


if __name__ == "__main__":
    main()
