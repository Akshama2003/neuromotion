"""Stage 3: joint angle computation from 3D landmarks.

Hip, knee and ankle angles are SIGNED sagittal-plane angles, computed in a body-forward
frame that is estimated from the clip itself (so they do not depend on which way the person
walks, or on which way the camera faces):

  hip_flexion        + flexion (thigh forward of trunk), - extension
  knee_flexion       + flexion, - hyperextension
  ankle_dorsiflexion + dorsiflexion (toes up), - plantarflexion
  trunk_lean         + forward lean, - backward lean

Elbow flexion and trunk_inclination stay unsigned 3D angles.
Assumes an upright camera (MediaPipe world y points down) and a roughly side-on view;
`sagittal_frame` reports how consistent the estimated forward direction is.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from pose import LM

UP = np.array([0.0, -1.0, 0.0])  # MediaPipe world: y points down


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


def _horizontal(v: np.ndarray) -> np.ndarray:
    return v - np.outer(v @ UP, UP)


def sagittal_frame(world: np.ndarray, visibility: np.ndarray | None = None,
                   min_vis: float = 0.3) -> dict:
    """Estimate the body-forward direction (unit vector, horizontal) for the whole clip.

    Primary: mean heel->toe direction of both feet (feet point forward when walking, standing,
    squatting). Cross-check: forward from the pelvis orientation. `agreement_deg` is the angle
    between the two; a large value suggests the camera is not side-on or landmarks are poor.
    """
    w = world.copy()
    if visibility is not None:
        w[visibility < min_vis] = np.nan

    def p(name):
        return w[:, LM[name], :]

    # pelvis-based forward (cross(up, lateral)); sign is convention-dependent
    lat = _horizontal(p("r_hip") - p("l_hip"))
    lat = lat[np.all(np.isfinite(lat), axis=1)]
    pelvis_fwd = None
    if len(lat):
        lat = lat / np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-9)
        pf = np.cross(UP, lat)
        m = pf.mean(axis=0)
        if np.linalg.norm(m) > 1e-6:
            pelvis_fwd = m / np.linalg.norm(m)

    # foot-based forward
    d = np.concatenate([_horizontal(p(f"{s}_foot") - p(f"{s}_heel")) for s in ("l", "r")], axis=0)
    d = d[np.all(np.isfinite(d), axis=1)]
    d = d[np.linalg.norm(d, axis=1) > 0.02]
    foot_fwd, consistency = None, 0.0
    if len(d):
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        m = d.mean(axis=0)
        consistency = float(np.linalg.norm(m))  # 1 = all feet point the same way
        if consistency > 0.3:
            foot_fwd = m / np.linalg.norm(m)

    if foot_fwd is not None:
        fwd, method = foot_fwd, "feet"
    elif pelvis_fwd is not None:
        fwd, method = pelvis_fwd, "pelvis"
    else:
        fwd, method = np.array([1.0, 0.0, 0.0]), "default"

    agreement = None
    if foot_fwd is not None and pelvis_fwd is not None:
        agreement = float(np.degrees(np.arccos(np.clip(abs(foot_fwd @ pelvis_fwd), 0, 1))))
    return {"forward": fwd, "method": method, "agreement_deg": agreement,
            "foot_consistency": consistency}


def _from_down(v, fwd):
    """Segment angle from straight-down, forward positive. v: (T,3) proximal->distal."""
    return np.degrees(np.arctan2(v @ fwd, -(v @ UP)))


def _from_up(v, fwd):
    """Segment angle from straight-up, forward positive."""
    return np.degrees(np.arctan2(v @ fwd, v @ UP))


def compute_joint_angles(world: np.ndarray, visibility: np.ndarray | None = None,
                         fps: float = 30.0, min_vis: float = 0.5,
                         smooth_cutoff: float | None = 6.0,
                         forward: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return a dict of angle time-series (degrees), each shape (T,)."""
    if forward is None:
        forward = sagittal_frame(world, visibility)["forward"]
    w = world.copy()
    if visibility is not None:
        w[visibility < min_vis] = np.nan  # drop low-confidence landmarks

    def p(name):
        return w[:, LM[name], :]

    hip_mid = (p("l_hip") + p("r_hip")) / 2
    sh_mid = (p("l_shoulder") + p("r_shoulder")) / 2
    trunk = sh_mid - hip_mid
    trunk_lean = _from_up(trunk, forward)

    out: dict[str, np.ndarray] = {}
    for s, side in (("l", "left"), ("r", "right")):
        thigh = _from_down(p(f"{s}_knee") - p(f"{s}_hip"), forward)
        shank = _from_down(p(f"{s}_ankle") - p(f"{s}_knee"), forward)
        foot_vec = p(f"{s}_foot") - p(f"{s}_heel")
        foot_pitch = np.degrees(np.arctan2(foot_vec @ UP, foot_vec @ forward))  # toe up = +
        out[f"hip_flexion_{side}"] = thigh + trunk_lean
        out[f"knee_flexion_{side}"] = thigh - shank
        out[f"ankle_dorsiflexion_{side}"] = foot_pitch - shank
        out[f"elbow_flexion_{side}"] = 180.0 - angle_3pt(
            p(f"{s}_shoulder"), p(f"{s}_elbow"), p(f"{s}_wrist"))

    out["trunk_lean"] = trunk_lean
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(np.sum(trunk * UP, -1) / np.linalg.norm(trunk, axis=-1), -1, 1)
    out["trunk_inclination"] = np.degrees(np.arccos(cos))  # unsigned, from vertical

    if smooth_cutoff:
        out = {k: smooth(v, fps, smooth_cutoff) for k, v in out.items()}
    return out
