"""Stage 6: structured movement report (JSON + Markdown)."""
from __future__ import annotations

import json
from datetime import datetime

DISCLAIMER = (
    "Research prototype. Outputs are movement measurements and heuristic observations for "
    "review by a qualified clinician or researcher. This is not a diagnosis or treatment advice."
)


def _fmt(v, nd=1):
    return "n/a" if v is None else f"{v:.{nd}f}"


def to_json(result: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def to_markdown(result: dict, source: str = "", figures: list[str] | None = None) -> str:
    j, g = result["joints"], result["gait"]
    lines = [
        "# NeuroMotion Movement Report", "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Source: {source or 'n/a'}",
        f"- Duration: {result['duration_s']:.1f} s ({result['frames']} frames @ {result['fps']:.0f} fps)",
        f"- Movement type (heuristic): **{result['movement_type_heuristic']}**", "",
        "## Observations", "",
    ]
    lines += [f"- {o}" for o in result["observations"]]
    lines += ["", "## Joint range of motion (degrees, 5th-95th percentile)", "",
              "| Joint | Left min | Left max | Left ROM | Right min | Right max | Right ROM |",
              "|---|---|---|---|---|---|---|"]
    for name in ("knee_flexion", "hip_flexion", "ankle_dorsiflexion", "elbow_flexion"):
        l, r = j[f"{name}_left"], j[f"{name}_right"]
        lines.append(f"| {name.replace('_', ' ')} | {_fmt(l['min'])} | {_fmt(l['max'])} | {_fmt(l['rom'])} "
                     f"| {_fmt(r['min'])} | {_fmt(r['max'])} | {_fmt(r['rom'])} |")
    t = j["trunk_inclination"]
    lines += ["", f"Trunk inclination from vertical: mean {_fmt(t['mean'])} deg, max {_fmt(t['max'])} deg.", "",
              "## Gait", "",
              f"- Steps: left {g['steps_left']}, right {g['steps_right']}",
              f"- Cadence: {g['cadence_steps_per_min']:.0f} steps/min",
              f"- Stride-time variability (CV): left {_fmt(g['stride_time_cv_pct']['left'])}%, "
              f"right {_fmt(g['stride_time_cv_pct']['right'])}%", "",
              "## Left/right asymmetry (symmetry index, %)", ""]
    for k, v in result["symmetry_index_pct"].items():
        lines.append(f"- {k.replace('_', ' ')}: {_fmt(v)}%")
    q = result.get("quality", {})
    if q:
        lines += ["", "## Data quality", "",
                  f"- Pose detected in {q['detection_rate']:.0%} of frames",
                  f"- Mean lower-body landmark visibility: {q['lower_body_visibility']:.2f}"]
    for fig in figures or []:
        lines += ["", f"![{fig}]({fig})"]
    lines += ["", "---", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)
