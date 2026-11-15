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
| `joint_angles.png` | knee, hip, ankle, elbow angles over time with detected steps, trunk inclination |
| `skeleton_frames.png` | side-view skeleton snapshots |
| `report.json` / `report.md` | ROM, gait (steps, cadence, stride variability), L/R symmetry, observations, data quality |
| `annotated.mp4` | video with skeleton and knee angles (with `--annotate`) |

## Layout

- `neuromotion/pose.py` - MediaPipe PoseLandmarker extraction (Tasks API; legacy `mp.solutions` is gone in current MediaPipe)
- `neuromotion/angles.py` - 3D joint angles + smoothing
- `neuromotion/analysis.py` - ROM, step detection, symmetry, heuristic observations (thresholds in `THRESHOLDS`)
- `neuromotion/viz.py`, `report.py`, `synthetic.py`
- `run.py` CLI, `app.py` Streamlit UI, `tests/` ground-truth tests on a synthetic skeleton

## Known limitations

- Monocular video: 3D world landmarks are estimates. Use a clear full-body side-on view, steady camera, good light.
- Angles are unsigned (0 = extended), so hip extension folds into the same range as flexion. Signed sagittal angles are a v1.1 item.
- Thresholds are prototype heuristics, not clinical norms. Output is measurement and observation, never a diagnosis.
- Not yet validated against real video or motion-capture ground truth.

## Next steps

1. Validate on real walking clips (and against a public gait dataset).
2. Signed sagittal-plane angles; heel-strike/toe-off gait events.
3. Stage 4 ML: classify movement patterns from the angle features (scikit-learn / PyTorch).
4. Stage 5: knee-joint robotic assistance in MuJoCo/OpenSim, driven by `report.json` and the angle series.
