#!/usr/bin/env python3
"""Draw a pose/identity overlay on a rendered clip (boxes + IDs + skeletons).

This is the "show me the computer vision" layer. The core pipeline stays
pure-Python and dependency-free; *this* module is an optional enrichment that
runs a real detection+pose model (Ultralytics YOLOv8-pose) over a short rendered
highlight clip and burns the annotations into the video:

  * a bounding box per person, with a tracker-assigned id (so each fighter keeps
    the same number across the clip), and
  * the 17-keypoint skeleton drawn on each body.

It is CPU-only (no GPU needed -- just slower) so it runs on GitHub's runners.
``ultralytics``/``opencv`` are imported lazily *inside* ``annotate`` so importing
this module never requires them; callers treat annotation as best-effort and
fall back to the un-annotated clip on any failure.
"""
import os
import subprocess

DEFAULT_MODEL = "yolov8n-pose.pt"   # nano pose model; auto-downloaded (~6.5 MB)
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640


def annotate(in_path, out_path=None, model=DEFAULT_MODEL, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
    """Annotate ``in_path`` with boxes + ids + skeletons, writing ``out_path``.

    ``out_path`` defaults to ``in_path`` (annotate in place). Uses YOLOv8-pose
    with ByteTrack so person ids persist across frames; ``Results.plot()`` draws
    the boxes, ids and skeleton. The annotated frames are written to a temporary
    AVI, then re-encoded to a browser-friendly H.264/yuv420p mp4 with ffmpeg.

    Returns the output path, or None if the model found no frames. Raises if the
    optional deps (ultralytics/opencv) or ffmpeg are missing -- callers catch it.
    """
    import cv2
    from ultralytics import YOLO

    out_path = out_path or in_path
    yolo = YOLO(model)

    fps = _source_fps(cv2, in_path)
    tmp = out_path + ".annot.avi"
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    try:
        # stream=True yields one Results per frame without loading the whole clip;
        # persist=True keeps tracker ids stable across the stream.
        for result in yolo.track(source=in_path, stream=True, persist=True,
                                 conf=conf, imgsz=imgsz, verbose=False):
            frame = result.plot()  # BGR ndarray with boxes/ids/skeleton drawn
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        return None  # no frames decoded

    # Re-encode browser-friendly (Safari/iOS need yuv420p + faststart).
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        check=True,
    )
    try:
        os.remove(tmp)
    except OSError:
        pass
    return out_path


def _source_fps(cv2, path, default=15.0):
    """Read a clip's frame rate (for the annotated writer), with a fallback."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return fps if fps and fps > 0 else default


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Draw pose/box overlay on a clip.")
    parser.add_argument("clip", help="Input video")
    parser.add_argument("--out", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = parser.parse_args()

    out = annotate(args.clip, out_path=args.out, model=args.model, conf=args.conf)
    print(f"Annotated -> {out}" if out else "No frames to annotate")


if __name__ == "__main__":
    main()
