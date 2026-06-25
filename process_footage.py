#!/usr/bin/env python3
"""Process one real recording end-to-end on GitHub (no Mac, no GPU).

This is the glue the weekly/dispatch workflow calls: take a real video (a path
in the repo or a URL to download), decode it to frames with ffmpeg, run the full
CV pipeline for the chosen sport domain, and publish browsable coaching outputs
under ``reports/<clip>/`` plus a single ``reports/index.json`` catalog.

The catalog is deliberately a flat JSON list of processed clips with relative
paths to each artifact -- exactly what a static phone PWA can fetch (from GitHub
Pages or the raw repo) to render a gallery of sessions, with no server to run.

Everything here is CPU-only: ffmpeg's software decoder + the pure-Python
pipeline. The only moving part that needs ffmpeg is the decode step, isolated in
decode_video.py.
"""
import json
import os
import re
from datetime import datetime, timezone

import decode_video
import highlights
import pipeline

DEFAULT_REPORTS_DIR = "reports"
INDEX_NAME = "index.json"
# Slugs too generic to make a useful session id (e.g. a Drive ".../view" tail);
# fall back to a timestamp so sessions stay distinct and readable.
_GENERIC_IDS = {"clip", "view", "uc", "download", "input-video", "input_video", "video"}


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name):
    """Filesystem/URL-safe id from a clip filename (stem only, lower-kebab)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-._").lower()
    return slug or "clip"


def update_index(reports_dir, entry):
    """Upsert ``entry`` (keyed by ``id``) into reports/index.json, newest first."""
    index_path = os.path.join(reports_dir, INDEX_NAME)
    try:
        with open(index_path) as fh:
            index = json.load(fh)
    except (FileNotFoundError, ValueError):
        index = {}
    clips = [c for c in index.get("clips", []) if c.get("id") != entry["id"]]
    clips.append(entry)
    clips.sort(key=lambda c: c.get("processed_at", ""), reverse=True)
    index = {"updated_at": _utc_now_iso(), "clips": clips}
    os.makedirs(reports_dir, exist_ok=True)
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return index_path


def process(
    src_video,
    domain,
    reports_dir=DEFAULT_REPORTS_DIR,
    fps=decode_video.DEFAULT_FPS,
    width=decode_video.DEFAULT_WIDTH,
    meters_per_pixel=None,
    source_label=None,
    name=None,
    work_dir=None,
):
    """Decode + run the pipeline for one video, publishing under reports/<id>/.

    Returns a summary dict (the catalog entry). ``src_video`` is a local path
    (already resolved/downloaded by the caller or decode_video.resolve_source).
    The tagged highlight clips are rendered with ffmpeg so the PWA can play them.
    Artifacts: ``reports/<id>/coaching/{report.json,summary.txt}``,
    ``reports/<id>/highlights/{manifest.json,<segment>.mp4}``,
    ``reports/<id>/results/metrics.json``.
    """
    clip_id = slugify(name or source_label or src_video)
    if clip_id in _GENERIC_IDS:
        clip_id = "clip-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    title = name or source_label or clip_id
    clip_dir = os.path.join(reports_dir, clip_id)
    work_dir = work_dir or clip_dir
    os.makedirs(work_dir, exist_ok=True)

    pgm_path = os.path.join(work_dir, f"{clip_id}.pgm.gz")
    decoded = decode_video.decode_to_pgm_gz(src_video, pgm_path, fps=fps, width=width)

    output_dir = os.path.join(clip_dir, "highlights")
    coaching_dir = os.path.join(clip_dir, "coaching")
    results_dir = os.path.join(clip_dir, "results")
    # video_path points the highlight ffmpeg commands at the real downloaded file
    # (tracking["source"] is just a display label).
    result = pipeline.run_pipeline(
        pgm_path,
        fps=fps,
        source=title,
        meters_per_pixel=meters_per_pixel,
        output_dir=output_dir,
        coaching_dir=coaching_dir,
        domain=domain,
        video_path=src_video,
    )
    pipeline._write_artifacts(result, output_dir, coaching_dir, results_dir)

    # Render the tagged highlight clips (ffmpeg). Best-effort: missing ffmpeg or a
    # single bad clip is skipped, never fatal. Record which actually rendered and
    # rewrite the manifest so the PWA only offers playable clips.
    manifest = result["manifest"]
    render_status = {r["id"]: r.get("status") for r in highlights.render_clips(manifest)}
    for clip in manifest["clips"]:
        clip["rendered"] = render_status.get(clip["id"]) == "rendered"
        clip.pop("ffmpeg_cmd", None)  # internal; not needed in the published manifest
    highlights.write_manifest(manifest, os.path.join(output_dir, "manifest.json"))

    # Decoded frames are an intermediate, not an artifact worth committing.
    try:
        os.remove(pgm_path)
    except OSError:
        pass

    metrics = result["metrics"]
    clips = [
        {
            "id": c["id"],
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "tags": c["tags"],
            "video": c["output"] if c.get("rendered") else None,
        }
        for c in manifest["clips"]
    ]
    entry = {
        "id": clip_id,
        "title": title,
        "domain": metrics["domain"],
        "source": result["tracking"].get("source"),
        "processed_at": _utc_now_iso(),
        "frames_processed": metrics["frames_processed"],
        "detected_frames": metrics["detected_frames"],
        "segment_count": metrics["segment_count"],
        "rendered_count": sum(1 for c in clips if c["video"]),
        "fps": fps,
        "frame_size": [decoded["width"], decoded["height"]],
        "report": os.path.join(clip_dir, "coaching", "report.json"),
        "summary": os.path.join(clip_dir, "coaching", "summary.txt"),
        "manifest": os.path.join(output_dir, "manifest.json"),
        "clips": clips,
    }
    update_index(reports_dir, entry)
    return entry


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decode + analyze one recording, publish reports.")
    parser.add_argument("src", nargs="?", help="Path to a local video file")
    parser.add_argument("--url", help="Download the video from this URL instead")
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--fps", type=float, default=decode_video.DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=decode_video.DEFAULT_WIDTH)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    parser.add_argument("--name", default=None, help="Session name (id/title); else derived from the file")
    args = parser.parse_args()

    src = decode_video.resolve_source(clip_path=args.src, clip_url=args.url)
    # Prefer the original filename (from a URL or path) as the human label/id,
    # stripping any query string so a Drive '...mp4?usp=...' tail stays clean.
    raw = args.src if args.src else (args.url or src)
    label = os.path.basename(raw.split("?", 1)[0].rstrip("/"))
    entry = process(
        src,
        domain=args.domain,
        reports_dir=args.reports_dir,
        fps=args.fps,
        width=args.width,
        meters_per_pixel=args.meters_per_pixel,
        source_label=label,
        name=args.name,
    )
    print(f"Processed {entry['source']} [{entry['domain']}]: "
          f"{entry['frames_processed']} frames -> {entry['segment_count']} segments. "
          f"Reports under {os.path.join(args.reports_dir, entry['id'])}/")


if __name__ == "__main__":
    main()
