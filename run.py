"""NeuroMotion v1 command line.

  python run.py --video walk.mp4 --out outputs/
  python run.py --landmarks outputs/landmarks.npz --out outputs/   # skip pose extraction
  python run.py --demo --out outputs/                              # synthetic walking
"""
import argparse
from pathlib import Path

from analysis import analyse
from gait import detect_gait_events
from angles import compute_joint_angles, sagittal_frame
from pose import PoseData, extract_pose
from report import to_json, to_markdown
from synthetic import make_walking
from viz import annotate_video, plot_angles, plot_skeleton_frames


def run(video=None, landmarks=None, demo=False, out="outputs", variant="full",
        annotate=False, asymmetric_demo=False, progress=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if demo:
        pose = make_walking(right_knee_scale=0.65 if asymmetric_demo else 1.0)
        source = "synthetic walking demo"
    elif landmarks:
        pose, source = PoseData.load(landmarks), str(landmarks)
    elif video:
        pose = extract_pose(video, variant=variant, progress=progress)
        pose.save(out / "landmarks.npz")
        source = str(video)
    else:
        raise SystemExit("Provide --video, --landmarks or --demo")

    frame = sagittal_frame(pose.world, pose.visibility)
    angles = compute_joint_angles(pose.world, pose.visibility, pose.fps, forward=frame["forward"])
    events = detect_gait_events(pose.world, pose.visibility, pose.fps, frame["forward"])
    result = analyse(angles, pose.fps, pose.visibility, pose.world, events=events, frame=frame)

    plot_angles(angles, pose.fps, result, out / "joint_angles.png", events=events)
    plot_skeleton_frames(pose.world, out / "skeleton_frames.png")
    to_json(result, out / "report.json")
    figs = ["joint_angles.png", "skeleton_frames.png"]
    (out / "report.md").write_text(to_markdown(result, source, figs))
    if annotate and video:
        annotate_video(video, pose, angles, out / "annotated.mp4")
    return pose, angles, result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--landmarks")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--asymmetric-demo", action="store_true", help="demo with a reduced right knee")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--variant", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--annotate", action="store_true", help="also write an annotated video")
    a = ap.parse_args()
    _, _, res = run(a.video, a.landmarks, a.demo, a.out, a.variant, a.annotate, a.asymmetric_demo)
    print(f"Done. Outputs in {a.out}/")
    print("Movement type:", res["movement_type_heuristic"])
    for o in res["observations"]:
        print(" -", o)
