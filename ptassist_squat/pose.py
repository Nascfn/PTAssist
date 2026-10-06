"""MediaPipe boundary: return plain normalized landmarks to the analyzer."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .landmarks import Landmark, Landmarks


NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

CONNECTIONS = (
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_shoulder", "right_hip"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"), ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
)


class PoseEstimator:
    def __init__(self, model_path: Path, min_detection: float = 0.5) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose model missing: {model_path}")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_detection,
            min_tracking_confidence=min_detection,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Landmarks:
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return {}
        return {
            name: Landmark(float(point.x), float(point.y),
                           float(point.visibility if point.visibility is not None else 0),
                           float(point.presence if point.presence is not None else 0))
            for name, point in zip(NAMES, result.pose_landmarks[0])
        }

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def draw_skeleton(frame: np.ndarray, landmarks: Landmarks, confidence: float = 0.4) -> None:
    height, width = frame.shape[:2]

    def position(name: str) -> tuple[int, int] | None:
        point = landmarks.get(name)
        if point is None or point.confidence < confidence:
            return None
        if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
            return None
        return round(point.x * width), round(point.y * height)

    for start, end in CONNECTIONS:
        a, b = position(start), position(end)
        if a and b:
            cv2.line(frame, a, b, (70, 220, 70), 2, cv2.LINE_AA)
    for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_heel", "right_heel", "left_foot_index", "right_foot_index"):
        point = position(name)
        if point:
            cv2.circle(frame, point, 4, (0, 220, 255), -1, cv2.LINE_AA)
