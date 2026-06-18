#!/usr/bin/env python3
"""Combined volleyball pipeline: local player tracking + Roboflow ball detection.

This is step 2 — the real pipeline, built on two validated pieces:
  * players + stable track IDs  -> local YOLOv8 (COCO `person`) + ByteTrack
  * the ball                    -> Roboflow-hosted volleyball model (HTTP)

For each frame it runs both, draws a combined overlay (player boxes + IDs, ball
marker + trail), and writes a single structured JSON of per-frame events — the
input the metrics layer and the Claude coaching call will consume.

The Roboflow call is the bottleneck (~3 fps over the network), so this is an
offline batch tool: run it, walk away, play back the result.

API key: pass --api-key, or (preferred — keeps it out of shell history) set it
in the environment first:

    export ROBOFLOW_API_KEY=your_key
    python3 poc/pipeline.py -i clip.mp4 --rf-model volleyball_detection/2

Use --no-ball to run players-only (no key needed) for a quick local check.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

PERSON_CLASS = 0  # COCO person
DETECT_URL = "https://detect.roboflow.com/{model}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Combined player-tracking + Roboflow ball-detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, type=Path, help="Input video clip.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Annotated video. Defaults to <input>_pipeline.mp4")
    p.add_argument("--json", type=Path, default=None,
                   help="Events JSON. Defaults to <input>_pipeline.json")
    p.add_argument("--player-model", default="yolov8n.pt",
                   help="Local YOLO weights for player detection/tracking.")
    p.add_argument("--rf-model", default="volleyball_detection/2",
                   help='Roboflow ball model id, "project-slug/version".')
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                   help="Roboflow Private API Key (or set ROBOFLOW_API_KEY env var).")
    p.add_argument("--player-conf", type=float, default=0.25, help="Player conf threshold.")
    p.add_argument("--ball-conf", type=float, default=0.30,
                   help="Ball conf threshold (0-1). Raise to cut false ball boxes.")
    p.add_argument("--imgsz", type=int, default=640, help="Player inference image size.")
    p.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N processed frames (0 = whole clip).")
    p.add_argument("--trail", type=int, default=20, help="Recent ball positions to draw.")
    p.add_argument("--no-ball", action="store_true",
                   help="Skip Roboflow ball detection (players only; no API key needed).")
    p.add_argument("--no-video", action="store_true", help="Skip annotated video (JSON only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import cv2
        from ultralytics import YOLO
        if not args.no_ball:
            import requests
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Run: pip install -r requirements.txt "
              "(and `pip install requests` for ball detection).", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.no_ball and not args.api_key:
        print("No Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY, "
              "or use --no-ball for players-only.", file=sys.stderr)
        return 2

    out_video = args.output or args.input.with_name(f"{args.input.stem}_pipeline.mp4")
    out_json = args.json or args.input.with_name(f"{args.input.stem}_pipeline.json")

    print(f"Player model: {args.player_model}")
    model = YOLO(args.player_model)
    if not args.no_ball:
        print(f"Ball model (Roboflow): {args.rf_model}")
        import requests
        session = requests.Session()
        rf_url = DETECT_URL.format(model=args.rf_model)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Could not open video: {args.input}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {w}x{h} @ {fps:.1f}fps")

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps / args.stride, (w, h))

    trail: deque = deque(maxlen=args.trail)
    events: list[dict] = []
    frame_idx, processed, ball_seen = -1, 0, 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.stride > 1 and frame_idx % args.stride != 0:
            continue

        players = _detect_players(model, frame, args.player_conf, args.imgsz)
        ball = None
        if not args.no_ball:
            ball = _detect_ball(session, rf_url, cv2, frame, args.api_key, args.ball_conf)
            if ball is not None:
                ball_seen += 1
                trail.append((int(ball["center"][0]), int(ball["center"][1])))

        events.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "ball": ball,
            "players": players,
        })

        if writer is not None:
            _draw(cv2, frame, players, ball, trail)
            writer.write(frame)

        processed += 1
        if processed % 10 == 0:
            rate = processed / (time.time() - t0)
            print(f"  {processed} frames | ball in {ball_seen} | {rate:.1f} fps", flush=True)
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    out_json.write_text(json.dumps({
        "source": str(args.input),
        "player_model": args.player_model,
        "ball_model": None if args.no_ball else args.rf_model,
        "fps": fps, "frame_size": [w, h], "stride": args.stride,
        "frames_processed": processed, "frames_with_ball": ball_seen,
        "events": events,
    }, indent=2))

    pct = (100.0 * ball_seen / processed) if processed else 0.0
    print(f"\nDone: {processed} frames in {time.time() - t0:.1f}s.")
    if not args.no_ball:
        print(f"Ball detected in {ball_seen}/{processed} frames ({pct:.0f}%).")
    print(f"  JSON:  {out_json}")
    if writer is not None:
        print(f"  Video: {out_video}")
    return 0


def _detect_players(model, frame, conf, imgsz) -> list[dict]:
    results = model.track(frame, persist=True, conf=conf, imgsz=imgsz,
                          classes=[PERSON_CLASS], verbose=False)
    r = results[0]
    players: list[dict] = []
    if r.boxes is None or len(r.boxes) == 0:
        return players
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    ids = (r.boxes.id.cpu().numpy().astype(int)
           if r.boxes.id is not None else [None] * len(confs))
    for box, c, tid in zip(xyxy, confs, ids):
        x1, y1, x2, y2 = (float(v) for v in box)
        players.append({"track_id": int(tid) if tid is not None else None,
                        "bbox": [x1, y1, x2, y2], "conf": float(c)})
    return players


def _detect_ball(session, url, cv2, frame, api_key, ball_conf):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("ascii")
    try:
        resp = session.post(
            url, params={"api_key": api_key, "confidence": int(ball_conf * 100)},
            data=b64, headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001 - network hiccup shouldn't kill the run
        print(f"  (ball request failed: {e})", file=sys.stderr)
        return None
    if resp.status_code != 200:
        print(f"  (Roboflow HTTP {resp.status_code}: {resp.text[:200]})", file=sys.stderr)
        return None
    best = None
    for d in resp.json().get("predictions", []):
        if "ball" in str(d.get("class", "")).lower():
            if best is None or d["confidence"] > best["confidence"]:
                best = d
    if best is None:
        return None
    return {"center": [best["x"], best["y"]],
            "bbox": [best["x"] - best["width"] / 2, best["y"] - best["height"] / 2,
                     best["x"] + best["width"] / 2, best["y"] + best["height"] / 2],
            "conf": float(best["confidence"])}


def _draw(cv2, frame, players, ball, trail) -> None:
    for pl in players:
        x1, y1, x2, y2 = (int(v) for v in pl["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"P{pl['track_id']}" if pl["track_id"] is not None else "player"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    pts = list(trail)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 2)
    if ball is not None:
        cx, cy = int(ball["center"][0]), int(ball["center"][1])
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"ball {ball['conf']:.2f}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
