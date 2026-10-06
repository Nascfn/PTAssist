"""Local annotated MP4 review clips with a synchronized feedback timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class ReviewClip:
    def __init__(self, measurement_key: str = "joint_errors_torso_lengths") -> None:
        self.frames: list[np.ndarray] = []
        self.timeline: list[dict] = []
        self.measurement_key = measurement_key

    def add(self, frame: np.ndarray, timestamp: float, feedback: str,
            confidence: float, errors: dict[str, float] | None = None) -> None:
        self.frames.append(frame.copy())
        self.timeline.append({"time_seconds": round(timestamp, 3),
                              "feedback": feedback, "confidence": round(confidence, 3),
                              self.measurement_key: {key: round(value, 3)
                                                     for key, value in (errors or {}).items()}})

    def save(self, path: Path, fps: float, verdict: dict) -> tuple[Path, Path]:
        if not self.frames:
            raise ValueError("No frames to save for review")
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create review video: {path}")
        try:
            for frame in self.frames:
                writer.write(frame)
        finally:
            writer.release()
        manifest = path.with_suffix(".json")
        manifest.write_text(json.dumps({"video": str(path), "fps": fps,
                                        "verdict": verdict, "timeline": self.timeline},
                                       indent=2)+"\n", encoding="utf-8")
        return path, manifest


def play_review(path: Path) -> None:
    """V opens a review; space pauses, arrows seek ~1 second, Q returns to live."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open review video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    window = "PTAssist review | Space pause | arrows seek | Q return"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    paused = False
    frame = None
    try:
        while True:
            if not paused:
                ok, next_frame = capture.read()
                if ok:
                    frame = next_frame
                else:
                    paused = True
            if frame is None:
                break
            shown = frame.copy()
            cv2.rectangle(shown, (0, shown.shape[0]-33),
                          (shown.shape[1], shown.shape[0]), (18, 24, 38), -1)
            cv2.putText(shown, "SPACE pause/play    LEFT/RIGHT seek    Q return",
                        (12, shown.shape[0]-11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (235, 245, 250), 1, cv2.LINE_AA)
            cv2.imshow(window, shown)
            key = cv2.waitKeyEx(0 if paused else max(1, round(1000/fps)))
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (2424832, 2555904):
                offset = -1 if key == 2424832 else 1
                at = capture.get(cv2.CAP_PROP_POS_FRAMES)
                capture.set(cv2.CAP_PROP_POS_FRAMES,
                            max(0, at+offset*round(fps)))
                ok, next_frame = capture.read()
                if ok:
                    frame = next_frame
    finally:
        capture.release()
        cv2.destroyWindow(window)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a local PTAssist annotated attempt")
    parser.add_argument("video", type=Path, help="Path to an attempt MP4")
    args = parser.parse_args()
    play_review(args.video)


if __name__ == "__main__":
    main()
