#!/usr/bin/env python3
"""Per-clip coaching reports: rally length, ball speed, contact-zone heatmap.

This is the project's core user-facing value: for each ingested clip, turn the
tracking data into a coaching-feedback summary a player or coach can actually
read. It builds directly on the tracking schema produced by detect.py and the
rally windows produced by highlights.segment_rallies, so it is verifiable
end-to-end against the bundled reference clip (see pipeline.py --self-test).

Three signals per clip (and per rally):

  * rally length    -- duration in seconds and the number of tracked contacts.
  * ball speed      -- average and peak speed from frame-to-frame ball motion,
                       in pixels/second, plus meters/second when a court
                       calibration (meters_per_pixel) is supplied.
  * contact heatmap -- a coarse grid over the court counting where coaching
                       contacts (serve/set/attack/...) happened, so hot zones
                       are visible at a glance.

Everything here is pure and stdlib-only: no rendering, no model calls.
"""
import json
import math
import os
from datetime import datetime, timezone

import highlights

DEFAULT_HEATMAP_COLS = 6
DEFAULT_HEATMAP_ROWS = 4
# ASCII ramp from empty to hot, used to render the heatmap in the text summary.
_HEAT_RAMP = " .:-=+*#%@"


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frames_in_window(frames, start, end, fps):
    """Tracked (ball-present) frames whose timestamp falls within [start, end]."""
    out = []
    for fr in frames:
        if fr.get("ball") is None:
            continue
        t = highlights._frame_time(fr, fps)
        if start <= t <= end:
            out.append((t, fr["ball"]))
    return out


def ball_speed(frames, start, end, fps, meters_per_pixel=None):
    """Average/peak ball speed across a window from frame-to-frame motion.

    Speed is measured between consecutive tracked positions as Euclidean pixel
    distance over elapsed time. When ``meters_per_pixel`` is given, metric
    speeds are added (clearly calibration-dependent, never fabricated when it
    isn't). Returns None when there isn't enough motion to measure.
    """
    samples = _frames_in_window(frames, start, end, fps)
    speeds = []
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        speeds.append(dist / dt)
    if not speeds:
        return None
    avg = sum(speeds) / len(speeds)
    peak = max(speeds)
    result = {
        "avg_px_per_s": round(avg, 2),
        "peak_px_per_s": round(peak, 2),
        "samples": len(speeds),
    }
    if meters_per_pixel:
        result["avg_m_per_s"] = round(avg * meters_per_pixel, 2)
        result["peak_m_per_s"] = round(peak * meters_per_pixel, 2)
    return result


def _ball_at(frames, t, fps):
    """Ball position of the tracked frame nearest in time to ``t`` (or None)."""
    best = None
    best_dt = None
    for fr in frames:
        if fr.get("ball") is None:
            continue
        dt = abs(highlights._frame_time(fr, fps) - t)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = fr["ball"]
    return best


def _cell_for(point, width, height, cols, rows):
    """Map a pixel point to a (col, row) grid cell, clamped to the grid."""
    x, y = point
    col = min(cols - 1, max(0, int(x / width * cols))) if width else 0
    row = min(rows - 1, max(0, int(y / height * rows))) if height else 0
    return col, row


def contact_heatmap(frames, events, width, height, fps, cols=DEFAULT_HEATMAP_COLS, rows=DEFAULT_HEATMAP_ROWS):
    """Bin coaching contacts into a ``rows x cols`` grid of counts.

    Each event is placed at the ball position nearest its timestamp (or its own
    ``pos`` if the event carries one), then counted into a court cell. Returns
    the grid (row-major list of lists), its dimensions, and the total binned.
    """
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    binned = 0
    for ev in events or []:
        t = ev.get("t")
        if t is None:
            continue
        point = ev.get("pos") or _ball_at(frames, t, fps)
        if point is None:
            continue
        col, row = _cell_for(point, width, height, cols, rows)
        grid[row][col] += 1
        binned += 1
    return {"cols": cols, "rows": rows, "grid": grid, "contacts_binned": binned}


def render_heatmap(heatmap):
    """Render a heatmap grid as ASCII rows (hottest cell = densest glyph)."""
    grid = heatmap["grid"]
    peak = max((max(row) for row in grid), default=0)
    lines = []
    for row in grid:
        chars = []
        for count in row:
            if peak == 0:
                chars.append(_HEAT_RAMP[0])
            else:
                level = int(round(count / peak * (len(_HEAT_RAMP) - 1)))
                chars.append(_HEAT_RAMP[level])
        lines.append("".join(chars))
    return lines


def rally_report(rally, index, tracking, meters_per_pixel=None):
    """Build the coaching report for a single rally window."""
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])
    fps = float(tracking.get("fps") or 30.0)
    start, end = rally["start"], rally["end"]
    contacts = [ev for ev in events if ev.get("t") is not None and start <= ev["t"] <= end]
    return {
        "id": f"rally_{index:03d}",
        "start": round(start, 3),
        "end": round(end, 3),
        "length_s": round(end - start, 3),
        "contacts": len(contacts),
        "tags": highlights.tag_rally(rally, events),
        "ball_speed": ball_speed(frames, start, end, fps, meters_per_pixel=meters_per_pixel),
    }


def build_report(
    tracking,
    rallies=None,
    meters_per_pixel=None,
    heatmap_cols=DEFAULT_HEATMAP_COLS,
    heatmap_rows=DEFAULT_HEATMAP_ROWS,
    max_gap_s=highlights.DEFAULT_MAX_GAP_S,
    min_rally_s=highlights.DEFAULT_MIN_RALLY_S,
):
    """Build a full coaching report for a clip's tracking data.

    Segments rallies (unless caller supplies them), reports per-rally length /
    speed / tags, and a clip-wide contact-zone heatmap plus roll-up totals.
    """
    fps = float(tracking.get("fps") or 30.0)
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])
    width = tracking.get("width") or 0
    height = tracking.get("height") or 0

    if rallies is None:
        rallies = highlights.segment_rallies(frames, fps, max_gap_s=max_gap_s, min_rally_s=min_rally_s)

    rally_reports = [rally_report(r, i, tracking, meters_per_pixel) for i, r in enumerate(rallies, start=1)]
    heatmap = contact_heatmap(frames, events, width, height, fps, cols=heatmap_cols, rows=heatmap_rows)

    total_play = round(sum(r["length_s"] for r in rally_reports), 3)
    speeds = [r["ball_speed"]["peak_px_per_s"] for r in rally_reports if r["ball_speed"]]
    overall_peak = max(speeds) if speeds else None

    return {
        "generated_at": _utc_now_iso(),
        "source": tracking.get("source"),
        "fps": fps,
        "court": {"width": width, "height": height, "meters_per_pixel": meters_per_pixel},
        "rally_count": len(rally_reports),
        "total_play_s": total_play,
        "longest_rally_s": max((r["length_s"] for r in rally_reports), default=0.0),
        "peak_ball_speed_px_per_s": overall_peak,
        "rallies": rally_reports,
        "contact_heatmap": heatmap,
    }


def render_summary(report):
    """Render a human-readable coaching summary (the coach-facing artifact)."""
    lines = []
    lines.append(f"Coaching report -- {report.get('source')}")
    lines.append(f"  rallies: {report['rally_count']}   total play: {report['total_play_s']}s"
                 f"   longest: {report['longest_rally_s']}s")
    peak = report.get("peak_ball_speed_px_per_s")
    if peak is not None:
        lines.append(f"  peak ball speed: {peak} px/s")
    lines.append("")
    for r in report["rallies"]:
        spd = r["ball_speed"]
        if spd:
            extra = f", peak {spd['peak_px_per_s']} px/s"
            if "peak_m_per_s" in spd:
                extra += f" ({spd['peak_m_per_s']} m/s)"
        else:
            extra = ", speed n/a"
        tags = ", ".join(r["tags"]) or "untagged"
        lines.append(f"  {r['id']}: {r['length_s']}s, {r['contacts']} contacts [{tags}]{extra}")
    lines.append("")
    lines.append("  contact-zone heatmap (court top-left origin):")
    for row in render_heatmap(report["contact_heatmap"]):
        lines.append(f"    |{row}|")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a coaching report from tracking JSON.")
    parser.add_argument("tracking", help="Path to tracking JSON (see detect.py / highlights.py)")
    parser.add_argument("--output-dir", default="coaching")
    parser.add_argument("--meters-per-pixel", type=float, default=None,
                        help="Court calibration to add metric ball speeds")
    args = parser.parse_args()

    with open(args.tracking) as fh:
        tracking = json.load(fh)

    report = build_report(tracking, meters_per_pixel=args.meters_per_pixel)
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "report.json")
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    summary = render_summary(report)
    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")
    print(summary)
    print(f"\nWrote {report_path} and {summary_path}")


if __name__ == "__main__":
    main()
