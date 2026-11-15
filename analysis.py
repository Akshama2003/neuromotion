"""Stage 3-4: movement metrics and deviation flags from joint-angle time-series.

Thresholds below are simple, configurable heuristics for a research prototype. They are
NOT clinically validated norms and the output is never a diagnosis.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .pose import LM

THRESHOLDS = {
    "asymmetry_pct": 15.0,        # side-to-side difference considered notable
    "min_swing_knee_deg": 40.0,   # walking: reduced peak knee flexion below this
    "trunk_lean_deg": 10.0,       # mean trunk inclination from vertical
    "stride_cv_pct": 15.0,        # stride-time variability
    "cadence_range": (80.0, 140.0),
    "min_detection_rate": 0.8,
}


def joint_stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return {"min": None, "max": None, "rom": None, "mean": None}
    lo, hi = np.percentile(x, [5, 95])
    return {"min": float(lo), "max": float(hi), "rom": float(hi - lo), "mean": float(x.mean())}


def detect_steps(knee_flexion: np.ndarray, fps: float) -> np.ndarray:
    """Swing-phase knee-flexion peaks, one per stride for the given leg."""
    x = np.nan_to_num(knee_flexion, nan=0.0)
    peaks, _ = find_peaks(x, prominence=15.0, distance=max(1, int(0.4 * fps)))
    return peaks


def symmetry_index(a, b) -> float | None:
    if a is None or b is None:
        return None
    m = 0.5 * (a + b)
    return None if m < 1e-6 else float(100.0 * abs(a - b) / m)


def _cv(intervals: np.ndarray) -> float | None:
    if len(intervals) < 3 or np.mean(intervals) <= 0:
        return None
    return float(100.0 * np.std(intervals) / np.mean(intervals))


def analyse(angles: dict[str, np.ndarray], fps: float, visibility: np.ndarray | None = None,
            world: np.ndarray | None = None) -> dict:
    T = len(next(iter(angles.values())))
    duration = T / fps
    th = THRESHOLDS

    # data quality
    quality = {}
    if visibility is not None:
        lower = [LM[n] for n in ("l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")]
        quality["detection_rate"] = float(np.mean(np.any(visibility > 0, axis=1)))
        quality["lower_body_visibility"] = float(np.mean(visibility[:, lower]))

    joints = {k: joint_stats(v) for k, v in angles.items()}

    # steps / gait
    peaks_l = detect_steps(angles["knee_flexion_left"], fps)
    peaks_r = detect_steps(angles["knee_flexion_right"], fps)
    n_steps = len(peaks_l) + len(peaks_r)
    cadence = n_steps / duration * 60.0 if duration > 0 else 0.0
    stride_cv = {
        "left": _cv(np.diff(peaks_l) / fps),
        "right": _cv(np.diff(peaks_r) / fps),
    }
    gait = {
        "steps_left": int(len(peaks_l)), "steps_right": int(len(peaks_r)),
        "cadence_steps_per_min": float(cadence),
        "stride_time_cv_pct": stride_cv,
        "step_frames_left": peaks_l.tolist(), "step_frames_right": peaks_r.tolist(),
    }

    # movement-type heuristic
    knee_rom = max((joints["knee_flexion_left"]["rom"] or 0), (joints["knee_flexion_right"]["rom"] or 0))
    if n_steps >= 4 and 50 <= cadence <= 200:
        movement = "walking"
    elif knee_rom > 60:
        movement = "squat / sit-stand like"
    elif knee_rom < 15:
        movement = "mostly static"
    else:
        movement = "other / mixed"

    # symmetry
    sym = {}
    for j in ("knee_flexion", "hip_flexion", "ankle_dorsiflexion"):
        sym[f"{j}_rom_pct"] = symmetry_index(joints[f"{j}_left"]["rom"], joints[f"{j}_right"]["rom"])
    sym["peak_knee_flexion_pct"] = symmetry_index(joints["knee_flexion_left"]["max"],
                                                  joints["knee_flexion_right"]["max"])

    # observations (for clinician/researcher review)
    obs = []
    if quality.get("detection_rate", 1.0) < th["min_detection_rate"]:
        obs.append(f"Low pose-detection rate ({quality['detection_rate']:.0%}); results may be unreliable. "
                   "Use a clear full-body, side-on view with good lighting.")
    for key, label in (("knee_flexion_rom_pct", "knee"), ("hip_flexion_rom_pct", "hip")):
        v = sym[key]
        if v is not None and v > th["asymmetry_pct"]:
            l, r = joints[f"{label}_flexion_left"]["rom"], joints[f"{label}_flexion_right"]["rom"]
            side = "left" if l < r else "right"
            obs.append(f"{label.capitalize()} range of motion differs {v:.0f}% between sides "
                       f"(L {l:.0f} deg, R {r:.0f} deg); {side} side is smaller.")
    if movement == "walking":
        for side in ("left", "right"):
            mx = joints[f"knee_flexion_{side}"]["max"]
            if mx is not None and mx < th["min_swing_knee_deg"]:
                obs.append(f"Reduced peak knee flexion on the {side} during walking ({mx:.0f} deg).")
        lo, hi = th["cadence_range"]
        if not lo <= cadence <= hi:
            obs.append(f"Cadence {cadence:.0f} steps/min is outside the typical {lo:.0f}-{hi:.0f} range.")
        for side, cv in stride_cv.items():
            if cv is not None and cv > th["stride_cv_pct"]:
                obs.append(f"High stride-time variability on the {side} (CV {cv:.0f}%).")
    lean = joints["trunk_inclination"]["mean"]
    if lean is not None and lean > th["trunk_lean_deg"]:
        obs.append(f"Mean trunk inclination {lean:.0f} deg from vertical.")
    if not obs:
        obs.append("No notable deviations against the prototype thresholds.")

    return {
        "duration_s": float(duration), "fps": float(fps), "frames": int(T),
        "movement_type_heuristic": movement,
        "quality": quality, "joints": joints, "gait": gait,
        "symmetry_index_pct": sym, "observations": obs,
        "thresholds": {k: (list(v) if isinstance(v, tuple) else v) for k, v in th.items()},
    }
