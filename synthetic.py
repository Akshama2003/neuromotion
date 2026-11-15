"""Synthetic walking skeleton so the pipeline can be tested without video or a model download."""
from __future__ import annotations

import numpy as np

from .pose import LM, PoseData


def make_walking(duration=10.0, fps=30.0, stride_hz=0.9, right_knee_scale=1.0,
                 noise=0.004, seed=0) -> PoseData:
    """Side-view walking in MediaPipe-like world coords (metres, y down, x forward)."""
    rng = np.random.default_rng(seed)
    T = int(duration * fps)
    t = np.arange(T) / fps
    thigh, shank, trunk, foot = 0.45, 0.43, 0.50, 0.18
    world = np.zeros((T, 33, 3))

    def leg(phase, knee_scale, z):
        ph = 2 * np.pi * stride_hz * t + phase
        theta = np.radians(25) * np.sin(ph)
        kappa = np.radians(8 + 52 * knee_scale * np.maximum(0, np.sin(ph + 0.3)) ** 2)
        beta = theta - kappa
        hip = np.stack([np.zeros(T), np.zeros(T), np.full(T, z)], 1)
        knee = hip + np.stack([thigh * np.sin(theta), thigh * np.cos(theta), np.zeros(T)], 1)
        ankle = knee + np.stack([shank * np.sin(beta), shank * np.cos(beta), np.zeros(T)], 1)
        toe = ankle + foot * np.stack([np.cos(beta), -np.sin(beta), np.zeros(T)], 1)
        heel = ankle - 0.05 * np.stack([np.cos(beta), -np.sin(beta), np.zeros(T)], 1)
        return hip, knee, ankle, toe, heel, ph

    lean = np.radians(5)
    trunk_vec = trunk * np.array([np.sin(lean), -np.cos(lean), 0.0])
    for s, phase, scale, z in (("l", 0.0, 1.0, -0.09), ("r", np.pi, right_knee_scale, 0.09)):
        hip, knee, ankle, toe, heel, ph = leg(phase, scale, z)
        sh = hip + trunk_vec
        arm = np.radians(20) * np.sin(ph + np.pi)
        elbow = sh + 0.28 * np.stack([np.sin(arm), np.cos(arm), np.zeros(T)], 1)
        fore = arm + np.radians(25)
        wrist = elbow + 0.26 * np.stack([np.sin(fore), np.cos(fore), np.zeros(T)], 1)
        for name, pts in (("hip", hip), ("knee", knee), ("ankle", ankle), ("foot", toe),
                          ("heel", heel), ("shoulder", sh), ("elbow", elbow), ("wrist", wrist)):
            world[:, LM[f"{s}_{name}"], :] = pts
    world += rng.normal(0, noise, world.shape)
    vis = np.full((T, 33), 0.99)
    image = np.zeros_like(world)
    image[..., 0] = 0.5 + world[..., 0] * 0.4
    image[..., 1] = 0.5 + world[..., 1] * 0.4
    return PoseData(fps=fps, image=image, world=world, visibility=vis,
                    width=640, height=480, source="synthetic")
