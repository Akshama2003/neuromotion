"""Stage 3: joint angle computation from 3D landmarks.

Angles are unsigned 3D angles between body segments, so they are independent of camera
orientation. Convention: 0 deg = fully extended / neutral, larger = more flexed.
Limitation: unsigned angles cannot distinguish hip extension from neutral.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .pose import LM


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b between segments b->a and b->c, in degrees. Inputs (T, 3)."""
    ba, bc = a - b, c - b
    num = np.sum(ba * bc, axis=-1)
    den = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def smooth(x: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    """Interpolate gaps then low-pass filter (zero-phase Butterworth)."""
    x = _interp_nan(x)
    if not np.all(np.isfinite(x)) or len(x) < 15:
        return x
    cutoff = min(cutoff, 0.4 * fps)
    b, a = butter(4, cutoff / (0.5 * fps))
    return filtfilt(b, a, x)


def compute_joint_angles(world: np.ndarray, visibility: np.ndarray | None = None,
                         fps: float = 30.0, min_vis: float = 0.5,
                         smooth_cutoff: float | None = 6.0) -> dict[str, np.ndarray]:
    """Return a dict of angle time-series (degrees), each shape (T,)."""
    w = world.copy()
    if visibility is not None:
        w[visibility < min_vis] = np.nan  # drop low-confidence landmarks

    def p(name):
        return w[:, LM[name], :]

    out: dict[str, np.ndarray] = {}
    for s, side in (("l", "left"), ("r", "right")):
        out[f"knee_flexion_{side}"] = 180.0 - angle_3pt(p(f"{s}_hip"), p(f"{s}_knee"), p(f"{s}_ankle"))
        out[f"hip_flexion_{side}"] = 180.0 - angle_3pt(p(f"{s}_shoulder"), p(f"{s}_hip"), p(f"{s}_knee"))
        out[f"ankle_dorsiflexion_{side}"] = 90.0 - angle_3pt(p(f"{s}_knee"), p(f"{s}_ankle"), p(f"{s}_foot"))
        out[f"elbow_flexion_{side}"] = 180.0 - angle_3pt(p(f"{s}_shoulder"), p(f"{s}_elbow"), p(f"{s}_wrist"))

    # Trunk inclination from vertical (MediaPipe world y points down => up is -y)
    hip_mid = (p("l_hip") + p("r_hip")) / 2
    sh_mid = (p("l_shoulder") + p("r_shoulder")) / 2
    trunk = sh_mid - hip_mid
    up = np.broadcast_to(np.array([0.0, -1.0, 0.0]), trunk.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(np.sum(trunk * up, -1) / np.linalg.norm(trunk, axis=-1), -1, 1)
    out["trunk_inclination"] = np.degrees(np.arccos(cos))

    if smooth_cutoff:
        out = {k: smooth(v, fps, smooth_cutoff) for k, v in out.items()}
    return out
