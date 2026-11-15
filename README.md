# NeuroMotion v1 - Human Body Understanding System

Video -> full-body skeleton -> joint angles -> movement plots -> structured report.
First stage of the NeuroMotion -> HealthGen Robotics roadmap.

## Quick start (local or Google Colab)

```bash
pip install -r requirements.txt
python run.py --demo --asymmetric-demo --out demo_output   # no video or model needed
python run.py --video my_walk.mp4 --out outputs/ --annotate  # real video
streamlit run app.py                                         # web UI
python -m pytest tests/ -q                                   # tests
```

In Colab: upload the folder, `!pip install -r requirements.txt`, then `!python run.py --video walk.mp4`.
The first real run downloads the MediaPipe pose model (~10 MB) to `~/.cache/neuromotion`.

## Outputs (in `--out`)

| File | Content |
|---|---|
| `landmarks.npz` | 33 landmarks per frame (image + 3D world coords + visibility); reusable via `--landmarks` |
| `joint_angles.png` | signed knee, hip, ankle, elbow angles, heel-strike/toe-off events, trunk lean |
| `skeleton_frames.png` | side-view skeleton snapshots |
| `report.json` / `report.md` | ROM, gait (steps, cadence, stride variability), L/R symmetry, observations, data quality |
| `annotated.mp4` | video with skeleton and knee angles (with `--annotate`) |

## Layout

- `pose.py` - MediaPipe PoseLandmarker extraction (Tasks API)
- `angles.py` - sagittal frame + signed joint angles + smoothing
- `gait.py` - heel strike / toe-off, stance/swing, step time and length
- `analysis.py` - ROM, symmetry, heuristic observations (thresholds in `THRESHOLDS`)
- `viz.py`, `report.py`, `synthetic.py`
- `run.py` CLI, `app.py` Streamlit UI, `tests/` tests on a synthetic skeleton

## Known limitations

- Monocular video: 3D world landmarks are estimates. Use a clear full-body side-on view, steady camera, good light.
- Hip, knee and ankle angles are signed sagittal-plane angles in a forward axis estimated from the feet (works for either walking direction). Needs an upright camera and a roughly side-on view; the report flags when foot and pelvis orientation disagree by more than 45 deg.
- Heel-strike / toe-off use the coordinate method (Zeni 2008) on monocular landmarks. The synthetic tests check logic, signs and invariance (mirroring, camera yaw, dropouts) but NOT timing accuracy on real gait; step length is approximate (monocular scale).
- Thresholds are prototype heuristics, not clinical norms. Output is measurement and observation, never a diagnosis.
- Not yet validated against real video or motion-capture ground truth.

## Next steps

1. Validate on real walking clips (and against a public gait dataset).
2. (done) Signed sagittal-plane angles; heel-strike/toe-off events, stance/swing, step time and length.
3. Stage 4 ML: classify movement patterns from the angle features (scikit-learn / PyTorch).
4. Stage 5: knee-joint robotic assistance in MuJoCo/OpenSim, driven by `report.json` and the angle series.
