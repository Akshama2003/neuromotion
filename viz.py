"""Stage 5: visualisation of movement over time."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pose import LM, SKELETON_EDGES

L_COLOR, R_COLOR = "#1f77b4", "#d62728"


def plot_angles(angles: dict, fps: float, analysis: dict | None, path) -> None:
    t = np.arange(len(next(iter(angles.values())))) / fps
    panels = [("knee_flexion", "Knee flexion (deg)"), ("hip_flexion", "Hip flexion (deg)"),
              ("ankle_dorsiflexion", "Ankle dorsiflexion (deg)"), ("elbow_flexion", "Elbow flexion (deg)")]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(10, 12), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        ax.plot(t, angles[f"{key}_left"], color=L_COLOR, label="Left")
        ax.plot(t, angles[f"{key}_right"], color=R_COLOR, label="Right")
        if key == "knee_flexion" and analysis:
            for side, c in (("left", L_COLOR), ("right", R_COLOR)):
                fr = np.array(analysis["gait"][f"step_frames_{side}"], dtype=int)
                if fr.size:
                    ax.plot(t[fr], angles[f"knee_flexion_{side}"][fr], "v", color=c, ms=7)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=2)
    axes[-1].plot(t, angles["trunk_inclination"], color="#2ca02c")
    axes[-1].set_ylabel("Trunk inclination (deg)")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].grid(alpha=0.3)
    fig.suptitle("NeuroMotion - joint angles over time (triangles = detected steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_skeleton_frames(world: np.ndarray, path, n: int = 6) -> None:
    """Side-view (x-y) skeleton snapshots across the clip."""
    T = world.shape[0]
    idx = np.linspace(0, T - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 4), sharex=True, sharey=True)
    for ax, i in zip(np.atleast_1d(axes), idx):
        for a, b in SKELETON_EDGES:
            pa, pb = world[i, LM[a]], world[i, LM[b]]
            color = L_COLOR if a.startswith("l_") else R_COLOR if a.startswith("r_") else "gray"
            ax.plot([pa[0], pb[0]], [-pa[1], -pb[1]], color=color, lw=2)
        ax.set_title(f"frame {i}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Skeleton snapshots (side view; blue = left, red = right)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def annotate_video(video_path, pose, angles, out_path) -> None:
    """Write a copy of the video with the skeleton and key angles drawn on it."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= pose.n_frames:
            break
        pts = pose.image[i]
        if np.all(np.isfinite(pts[:, :2])):
            xy = {k: (int(pts[v, 0] * w), int(pts[v, 1] * h)) for k, v in LM.items()}
            for a, b in SKELETON_EDGES:
                col = (255, 120, 30) if a.startswith("l_") else (40, 40, 220)
                cv2.line(frame, xy[a], xy[b], col, 3)
            for k, (x, y) in xy.items():
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            for side, key in (("l", "left"), ("r", "right")):
                ang = angles[f"knee_flexion_{key}"][i]
                cv2.putText(frame, f"{ang:.0f}", (xy[f"{side}_knee"][0] + 8, xy[f"{side}_knee"][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    _to_h264(out_path)


def _to_h264(path) -> None:
    """Re-encode to H.264 so browsers can play it. Silently keeps mp4v if ffmpeg is missing."""
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("ffmpeg"):
        return
    path = Path(path)
    tmp = path.with_name(path.stem + "_h264.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        capture_output=True)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(path)
