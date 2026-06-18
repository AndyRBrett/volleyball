#!/usr/bin/env python3
"""Build the prompt for the Claude coaching readout from a metrics dict.

Kept separate from the web app so the prompt logic is testable without an API
key. The actual API call (anthropic SDK, claude-opus-4-8) lives in webapp/app.py.

The summary is deliberately compact (aggregate stats, not per-frame events) and
honest about units/limitations so Claude grounds its analysis in what the data
supports instead of inventing technique it cannot see from stats alone.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are an experienced volleyball coach and performance analyst. You are "
    "given automatically-extracted statistics from a SINGLE short video clip — "
    "no scoreboard, no player names, no technique footage, just tracking-derived "
    "numbers. Ground every statement in the provided numbers. Do not invent "
    "details the data cannot support (you cannot judge form, scores, or who won). "
    "Be concrete and useful to a coach or player. Where a number is unreliable, "
    "say so plainly rather than over-interpreting it.\n\n"
    "Write a tight markdown readout with these sections:\n"
    "## Summary — 2-3 sentences on what this clip contained.\n"
    "## What the data shows — bullet points tied to specific numbers.\n"
    "## Coaching observations — what the patterns suggest (hedged appropriately).\n"
    "## Suggested focus — 2-4 concrete, general drills/next steps.\n"
    "## Data caveats — the limitations below, in one or two lines.\n"
    "Keep it concise — a coach should read it in under a minute."
)


def summarize(metrics: dict) -> dict:
    """Compact, prompt-ready summary of a metrics.json dict."""
    fps = metrics.get("fps") or 30.0
    stride = metrics.get("stride") or 1
    frames = metrics.get("frames_processed", 0)
    duration_s = round(frames * stride / fps, 1) if fps else None

    ball = metrics.get("ball", {})
    rallies = metrics.get("rallies", [])
    players = metrics.get("players", {})
    cal = metrics.get("calibration")
    pc = metrics.get("players_court")

    summary = {
        "clip": {"approx_duration_s": duration_s, "frames_sampled": frames,
                 "fps": fps, "sampled_every_n_frames": stride},
        "rallies": {"count": metrics.get("rally_count", len(rallies)),
                    "durations_s": [r.get("duration_s") for r in rallies]},
        "ball": {
            "detection_rate": ball.get("raw_detection_rate"),
            "avg_speed_px_per_s": ball.get("avg_speed_px_s"),
            "max_speed_px_per_s": ball.get("max_speed_px_s"),
            "note": "Ball speed is in PIXELS/second (relative, not real). Real "
                    "mph/kmh is unavailable — the ball is airborne and only the "
                    "ground plane is calibrated. Use it only for relative comparison.",
        },
        "players_uncalibrated": {
            "distinct_track_ids": players.get("track_count"),
            "note": "This count is unreliable: it includes background people / "
                    "other courts and inflates from tracker ID switches.",
        },
    }

    if cal and pc:
        summary["players_on_court_calibrated"] = {
            "court": cal.get("court"),
            "approximate": cal.get("approximate"),
            "players_on_court": pc.get("track_count_in_court"),
            "per_player": [
                {"id": p.get("track_id"), "distance_m": p.get("distance_m"),
                 "top_speed_kmh": p.get("top_speed_kmh"),
                 "top_speed_mph": p.get("top_speed_mph")}
                for p in pc.get("per_track", [])
            ],
            "note": ("Real-world units (metres, km/h, mph), accurate for a fixed "
                     "camera. APPROXIMATE here — the camera moved during the clip."
                     if cal.get("approximate") else
                     "Real-world units (metres, km/h, mph) from a fixed-camera "
                     "calibration."),
        }
    else:
        summary["players_on_court_calibrated"] = (
            "Not calibrated — no real-world units or court filtering available.")

    return summary


def build_user_message(summary: dict) -> str:
    return (
        "Here are the extracted statistics for one volleyball clip. Read the "
        "embedded notes about units and reliability carefully and reflect them in "
        "your analysis.\n\n```json\n" + json.dumps(summary, indent=2) + "\n```"
    )
