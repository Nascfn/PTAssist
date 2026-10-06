"""Compact, local video previews for automatically segmented teaching reps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TeachingVideo:
    frames: list[tuple[float, bytes]] = field(default_factory=list)

    def _decode(self, index: int) -> np.ndarray:
        frame = cv2.imdecode(np.frombuffer(self.frames[index][1], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Could not decode a teaching video frame")
        return frame

    def save(self, path: Path) -> Path:
        if not self.frames:
            raise ValueError("Teaching video has no frames")
        path.parent.mkdir(parents=True, exist_ok=True)
        first = self._decode(0)
        height, width = first.shape[:2]
        elapsed = self.frames[-1][0]-self.frames[0][0]
        fps = min(60.0, max(1.0, (len(self.frames)-1)/elapsed)) if elapsed > 0 else 15.0
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to save teaching video: {path}")
        try:
            writer.write(first)
            for i in range(1, len(self.frames)):
                writer.write(self._decode(i))
        finally:
            writer.release()
        return path


@dataclass
class TeachingVideoBuffer:
    """Keep short JPEG clips, discarding walking and idle frames immediately."""

    recent: deque[tuple[float, bytes]] = field(default_factory=deque)
    active: list[tuple[float, bytes]] = field(default_factory=list)
    candidates: list[TeachingVideo] = field(default_factory=list)

    def update(self, frame: np.ndarray, timestamp: float, events: list[dict]) -> None:
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError("Unable to encode teaching review frame")
        item = (timestamp, encoded.tobytes())
        self.recent.append(item)
        while self.recent and timestamp-self.recent[0][0] > 0.4:
            self.recent.popleft()
        started = False
        for event in events:
            if event["event"] == "candidate_started":
                self.active = list(self.recent)
                started = True
            elif event["event"] == "candidate_discarded":
                self.active = []
            elif event["event"] == "candidate_completed":
                if not started:
                    self.active.append(item)
                self.candidates.append(TeachingVideo(self.active))
                self.active = []
        if self.active and not started and not any(
                event["event"] == "candidate_completed" for event in events):
            self.active.append(item)


def play_teaching_video(video: TeachingVideo, number: int, total: int) -> None:
    """Show video plus pose skeleton; Space pauses, arrows seek, Q returns."""
    if not video.frames:
        raise ValueError("Teaching video has no frames")
    window = "PTAssist teaching review | Q returns to approval"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 675)
    elapsed = video.frames[-1][0]-video.frames[0][0]
    fps = min(60.0, max(1.0, (len(video.frames)-1)/elapsed)) if elapsed > 0 else 15.0
    index = 0
    paused = False
    try:
        while True:
            shown = video._decode(index)
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 42), (23, 29, 35), -1)
            cv2.putText(shown, f"CANDIDATE {number}/{total} | {index+1}/{len(video.frames)}",
                        (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (230, 245, 250), 2, cv2.LINE_AA)
            cv2.rectangle(shown, (0, shown.shape[0]-37),
                          (shown.shape[1], shown.shape[0]), (23, 29, 35), -1)
            cv2.putText(shown, "SPACE pause | arrows step | R replay | Q return",
                        (12, shown.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 245, 250), 1, cv2.LINE_AA)
            cv2.imshow(window, shown)
            key = cv2.waitKeyEx(0 if paused else max(1, round(1000/fps)))
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("r"), ord("R")):
                index, paused = 0, False
            elif key in (2424832, 2555904):
                index = max(0, min(len(video.frames)-1,
                                   index+(-1 if key == 2424832 else 1)))
                paused = True
            elif not paused:
                index += 1
                if index >= len(video.frames):
                    index, paused = len(video.frames)-1, True
    finally:
        cv2.destroyWindow(window)
