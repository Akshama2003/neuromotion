"""Tests on a synthetic skeleton with known ground truth.   python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angles import angle_3pt, compute_joint_angles, sagittal_frame
from gait import detect_gait_events, gait_metrics
from pose import PoseData
from run import run
from synthetic import make_walking


def _angles(pose, **kw):
    fr = sagittal_frame(pose.world, pose.visibility)
    return compute_joint_angles(pose.world, pose.visibility, pose.fps, forward=fr["forward"], **kw), fr


def test_angle_math():
    a, b, c = np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 0]]), np.array([[0.0, 1, 0]])
    assert abs(angle_3pt(a, b, c)[0] - 90) < 1e-6
    assert abs(angle_3pt(a, b, -a)[0] - 180) < 1e-6


def test_symmetric_walking(tmp_path):
    _, _, res = run(demo=True, out=tmp_path)
    assert res["movement_type_heuristic"] == "walking"
    assert res["gait"]["event_source"] == "heel_strike_toe_off"
    assert abs(res["gait"]["cadence_steps_per_min"] - 108) < 6      # 0.9 Hz x 2 legs
    assert abs(res["gait"]["stride_time_s"]["left"] - 1 / 0.9) < 0.05
    assert res["joints"]["knee_flexion_left"]["max"] > 45
    assert res["symmetry_index_pct"]["knee_flexion_rom_pct"] < 10
    assert "No notable deviations" in res["observations"][0], res["observations"]
    for f in ("report.json", "report.md", "joint_angles.png", "skeleton_frames.png"):
        assert (tmp_path / f).exists()
    json.loads((tmp_path / "report.json").read_text())


def test_asymmetry_is_flagged(tmp_path):
    _, _, res = run(demo=True, asymmetric_demo=True, out=tmp_path)
    assert any("Knee range of motion differs" in o and "right side is smaller" in o
               for o in res["observations"])


def test_knee_angle_matches_ground_truth():
    pose = make_walking(noise=0.0)
    ang, _ = _angles(pose, smooth_cutoff=None)
    assert abs(np.nanmax(ang["knee_flexion_left"]) - 60) < 1.0
    assert abs(np.nanmin(ang["knee_flexion_left"]) - 8) < 1.0


def test_hip_is_signed():
    """Synthetic thigh swings +/-25 deg about vertical, trunk leans 5 deg forward => hip -20..+30."""
    pose = make_walking(noise=0.0)
    ang, fr = _angles(pose, smooth_cutoff=None)
    assert fr["method"] == "feet"
    assert abs(np.nanmax(ang["hip_flexion_left"]) - 30) < 1.0
    assert abs(np.nanmin(ang["hip_flexion_left"]) - (-20)) < 1.0      # extension is negative
    assert abs(np.nanmean(ang["trunk_lean"]) - 5) < 0.5               # forward lean positive


def test_ankle_neutral_is_near_zero():
    pose = make_walking(noise=0.0)
    ang, _ = _angles(pose, smooth_cutoff=None)
    assert np.nanmax(np.abs(ang["ankle_dorsiflexion_left"])) < 1.0


def test_walking_direction_invariance():
    """Mirroring the world x-axis (walking the other way) must not change signed angles."""
    pose = make_walking(noise=0.0)
    a1, _ = _angles(pose, smooth_cutoff=None)
    mirrored = pose.world.copy()
    mirrored[..., 0] *= -1
    pose2 = PoseData(fps=pose.fps, image=pose.image, world=mirrored, visibility=pose.visibility)
    a2, _ = _angles(pose2, smooth_cutoff=None)
    for k in ("hip_flexion_left", "knee_flexion_right", "trunk_lean", "ankle_dorsiflexion_left"):
        assert np.allclose(a1[k], a2[k], atol=0.5), k


def test_camera_yaw_invariance():
    """Rotating the skeleton about the vertical axis (camera not exactly side-on) keeps sagittal angles."""
    pose = make_walking(noise=0.0)
    a1, _ = _angles(pose, smooth_cutoff=None)
    th = np.radians(25)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    rot = pose.world @ R.T
    pose2 = PoseData(fps=pose.fps, image=pose.image, world=rot, visibility=pose.visibility)
    a2, _ = _angles(pose2, smooth_cutoff=None)
    assert np.allclose(a1["knee_flexion_left"], a2["knee_flexion_left"], atol=0.5)
    assert np.allclose(a1["hip_flexion_right"], a2["hip_flexion_right"], atol=0.5)


def test_gait_events_alternate_and_count():
    pose = make_walking(duration=12, noise=0.003)
    fr = sagittal_frame(pose.world, pose.visibility)
    ev = detect_gait_events(pose.world, pose.visibility, pose.fps, fr["forward"])
    gm = gait_metrics(ev, pose.fps)
    for side in ("left", "right"):
        assert abs(gm["n_heel_strikes"][side] - 0.9 * 12) <= 1.5
        assert abs(gm["n_toe_offs"][side] - 0.9 * 12) <= 1.5
    seq = sorted([(i, s) for s in ("left", "right") for i in gm["heel_strike_frames"][s]])
    assert all(a[1] != b[1] for a, b in zip(seq[:-1], seq[1:])), "heel strikes should alternate L/R"
    assert gm["stance_pct"]["left"] is not None and 20 < gm["stance_pct"]["left"] < 80
    assert abs(gm["swing_pct"]["left"] + gm["stance_pct"]["left"] - 100) < 1e-6
    assert abs(gm["step_time_s"]["left"] - 1 / 1.8) < 0.06


def test_survives_dropouts_and_low_visibility(tmp_path):
    pose = make_walking(duration=10, noise=0.01)
    rng = np.random.default_rng(1)
    pose.world[rng.random(pose.n_frames) < 0.1] = np.nan           # 10% of frames lost
    pose.visibility[rng.random(pose.visibility.shape) < 0.1] = 0.1  # random low-confidence landmarks
    pose.visibility[pose.n_frames // 2:pose.n_frames // 2 + 20] = 0
    p = tmp_path / "lm.npz"
    pose.save(p)
    _, _, res = run(landmarks=p, out=tmp_path)
    assert res["movement_type_heuristic"] == "walking"
    assert 90 < res["gait"]["cadence_steps_per_min"] < 125
    json.loads((tmp_path / "report.json").read_text())


def test_static_and_empty_inputs_do_not_crash(tmp_path):
    static = make_walking(duration=5, noise=0.002)
    static.world[:] = static.world[0]                              # nobody moves
    p = tmp_path / "static.npz"
    static.save(p)
    _, _, res = run(landmarks=p, out=tmp_path / "a")
    assert res["movement_type_heuristic"] == "mostly static"
    assert res["gait"]["steps_left"] == 0

    nan = make_walking(duration=3)
    nan.world[:] = np.nan
    nan.visibility[:] = 0
    p2 = tmp_path / "nan.npz"
    nan.save(p2)
    run(landmarks=p2, out=tmp_path / "b")                           # must not raise
    assert (tmp_path / "b" / "report.md").exists()
