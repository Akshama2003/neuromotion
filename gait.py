"""Stage 4: gait events and spatiotemporal metrics.

Coordinate-based method (Zeni et al., 2008): along the body-forward axis, heel strike is the
most-forward heel position relative to the pelvis, and toe-off is the most-backward toe position.
Prototype quality: monocular video, no force plates, not clinically validated.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from angles import smooth
from pose import LM

SIDES = (("l", "left"), ("r", "right"))
MIN_EXCURSION_M = 0.05  # foot must move at least this far relative to the pelvis


def foot_signals(world, visibility, forward, fps, min_vis=0.3, cutoff=6.0) -> dict:
    """Forward position (m) of heel and toe relative to the pelvis, per side, smoothed."""
    w = world.copy()
    if visibility is not None:
        w[visibility < min_vis] = np.nan
    pelvis = (w[:, LM["l_hip"]] + w[:, LM["r_hip"]]) / 2
    sig = {}
    for s, side in SIDES:
        sig[f"heel_{side}"] = smooth((w[:, LM[f"{s}_heel"]] - pelvis) @ forward, fps, cutoff)
        sig[f"toe_{side}"] = smooth((w[:, LM[f"{s}_foot"]] - pelvis) @ forward, fps, cutoff)
    return sig


def _peaks(x: np.ndarray, distance: int) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.array([], dtype=int)
    rng = np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
    if rng < MIN_EXCURSION_M:
        return np.array([], dtype=int)
    filled = np.where(ok, x, np.nanmin(x))
    p, _ = find_peaks(filled, prominence=0.3 * rng, distance=distance)
    return p


def detect_gait_events(world, visibility, fps, forward, min_stride_s: float = 0.5) -> dict:
    sig = foot_signals(world, visibility, forward, fps)
    dist = max(1, int(min_stride_s * fps))
    ev = {"signals": sig, "heel_strikes": {}, "toe_offs": {}}
    for _, side in SIDES:
        ev["heel_strikes"][side] = _peaks(sig[f"heel_{side}"], dist)
        ev["toe_offs"][side] = _peaks(-sig[f"toe_{side}"], dist)
    return ev


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _cv(v):
    if len(v) < 3 or np.mean(v) <= 0:
        return None
    return float(100.0 * np.std(v) / np.mean(v))


def gait_metrics(ev: dict, fps: float) -> dict:
    """JSON-safe spatiotemporal metrics from detected events."""
    hs = {s: np.asarray(ev["heel_strikes"][s], dtype=int) for _, s in SIDES}
    to = {s: np.asarray(ev["toe_offs"][s], dtype=int) for _, s in SIDES}
    sig = ev["signals"]

    seq = sorted([(int(i), s) for s in hs for i in hs[s]])
    cadence = None
    if len(seq) >= 3:
        span = (seq[-1][0] - seq[0][0]) / fps
        if span > 0:
            cadence = float((len(seq) - 1) / span * 60.0)

    step_t = {"left": [], "right": []}
    step_l = {"left": [], "right": []}
    for (i, a), (j, b) in zip(seq[:-1], seq[1:]):
        dt = (j - i) / fps
        if a != b and 0.2 <= dt <= 1.5:
            step_t[b].append(dt)
            d = sig[f"heel_{b}"][j] - sig[f"heel_{a}"][j]
            if np.isfinite(d):
                step_l[b].append(abs(float(d)))

    stride, stance = {}, {}
    for _, s in SIDES:
        d = np.diff(hs[s]) / fps
        stride[s] = d[(d >= 0.6) & (d <= 2.5)]
        st = []
        for i, j in zip(hs[s][:-1], hs[s][1:]):
            T = (j - i) / fps
            if 0.6 <= T <= 2.5:
                t_off = to[s][(to[s] > i) & (to[s] < j)]
                if len(t_off):
                    st.append(100.0 * (t_off[0] - i) / (j - i))
        stance[s] = st

    stance_mean = {s: _mean(stance[s]) for s in stance}
    return {
        "n_heel_strikes": {s: int(len(hs[s])) for s in hs},
        "n_toe_offs": {s: int(len(to[s])) for s in to},
        "heel_strike_frames": {s: hs[s].tolist() for s in hs},
        "toe_off_frames": {s: to[s].tolist() for s in to},
        "cadence_steps_per_min": cadence,
        "stride_time_s": {s: _mean(stride[s]) for s in stride},
        "stride_time_cv_pct": {s: _cv(stride[s]) for s in stride},
        "step_time_s": {s: _mean(step_t[s]) for s in step_t},
        "step_length_m": {s: _mean(step_l[s]) for s in step_l},
        "stance_pct": stance_mean,
        "swing_pct": {s: (None if v is None else 100.0 - v) for s, v in stance_mean.items()},
    }
