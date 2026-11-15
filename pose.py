"""Stage 1-2: video -> full-body skeleton (33 MediaPipe landmarks per frame).

Uses the MediaPipe Tasks `PoseLandmarker` (the legacy `mp.solutions.pose` was removed
in recent MediaPipe releases). The model file is downloaded once and cached.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MediaPipe BlazePose landmark indices (33-point topology)
LM = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_foot": 31, "r_foot": 32,
}

# Bones used for drawing skeletons
SKELETON_EDGES = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_foot"), ("l_ankle", "l_heel"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_foot"), ("r_ankle", "r_heel"),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


@dataclass
class PoseData:
    """Per-frame landmarks. Missing detections are NaN."""

    fps: float
    image: np.ndarray       # (T, 33, 3) normalised image coords (x, y in 0..1, z relative)
    world: np.ndarray       # (T, 33, 3) metric coords in metres, origin at hip centre
    visibility: np.ndarray  # (T, 33)
    width: int = 0
    height: int = 0
    source: str = ""

    @property
    def n_frames(self) -> int:
        return self.world.shape[0]

    def save(self, path) -> None:
        np.savez_compressed(
            path, fps=self.fps, image=self.image, world=self.world,
            visibility=self.visibility, width=self.width, height=self.height,
            source=self.source,
        )

    @staticmethod
    def load(path) -> "PoseData":
        d = np.load(path, allow_pickle=False)
        return PoseData(
            fps=float(d["fps"]), image=d["image"], world=d["world"],
            visibility=d["visibility"], width=int(d["width"]),
            height=int(d["height"]), source=str(d["source"]),
        )


def ensure_model(variant: str = "full", cache_dir=None) -> Path:
    """Download the PoseLandmarker model if it is not cached. variant: lite|full|heavy."""
    cache = Path(cache_dir or Path.home() / ".cache" / "neuromotion")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"pose_landmarker_{variant}.task"
    if not path.exists():
        url = MODEL_URL.format(variant=variant)
        print(f"Downloading pose model ({variant}) ...")
        urllib.request.urlretrieve(url, path)
    return path


def extract_pose(video_path, variant: str = "full", max_frames: int | None = None,
                 model_path=None, progress=None) -> PoseData:
    """Run MediaPipe pose estimation on every frame of a video."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = Path(model_path) if model_path else ensure_model(variant)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    img_list, world_list, vis_list = [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(round(i * 1000.0 / fps)))
            if res.pose_landmarks and res.pose_world_landmarks:
                lm = res.pose_landmarks[0]
                wl = res.pose_world_landmarks[0]
                img_list.append([[p.x, p.y, p.z] for p in lm])
                world_list.append([[p.x, p.y, p.z] for p in wl])
                vis_list.append([p.visibility for p in lm])
            else:
                img_list.append(np.full((33, 3), np.nan))
                world_list.append(np.full((33, 3), np.nan))
                vis_list.append(np.zeros(33))
            i += 1
            if progress and total:
                progress(min(i / total, 1.0))
    cap.release()

    return PoseData(
        fps=float(fps),
        image=np.asarray(img_list, dtype=float),
        world=np.asarray(world_list, dtype=float),
        visibility=np.asarray(vis_list, dtype=float),
        width=width, height=height, source=str(video_path),
    )
