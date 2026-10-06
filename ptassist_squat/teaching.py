"""Separate stationary exercise repetitions from walk-up setup and pauses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .custom import PoseSample


MOTION_JOINTS = (
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
)


@dataclass(frozen=True)
class TeachingConfig:
    min_walk_away_seconds: float = 2.0
    settle_seconds: float = 0.85
    stable_center_torso_lengths: float = 0.22
    stable_scale_fraction: float = 0.09
    stable_joint_torso_lengths: float = 0.19
    zone_center_torso_lengths: float = 0.55
    zone_scale_fraction: float = 0.24
    start_motion_torso_lengths: float = 0.28
    return_motion_torso_lengths: float = 0.15
    min_peak_motion_torso_lengths: float = 0.40
    return_hold_seconds: float = 0.25
    min_rep_seconds: float = 0.8
    max_rep_seconds: float = 15.0
    max_missing_seconds: float = 0.65


@dataclass
class TeachingCapture:
    config: TeachingConfig = field(default_factory=TeachingConfig)
    state: str = "SETTLING"
    candidates: list[list[PoseSample]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    started_at: float | None = None
    last_valid_time: float | None = None
    baseline: dict[str, tuple[float, float]] = field(default_factory=dict)
    baseline_center: tuple[float, float] | None = None
    baseline_scale: float = 0.0
    recent: deque[PoseSample] = field(default_factory=deque)
    active: list[PoseSample] = field(default_factory=list)
    peak_motion: float = 0.0
    departure_streak: int = 0
    return_since: float | None = None

    @property
    def prompt(self) -> str:
        if self.state == "SETTLING":
            return "Walk back, then pause in your starting pose"
        if self.state == "READY":
            return f"Ready: do a rep and return ({len(self.candidates)} found)"
        return f"Tracking rep {len(self.candidates)+1}: return to start"

    def _settle(self, timestamp: float, reason: str) -> None:
        if self.state == "MOVING":
            self.events.append({"time_seconds": round(timestamp, 3),
                                "event": "candidate_discarded", "reason": reason})
        self.state = "SETTLING"
        self.baseline = {}
        self.baseline_center = None
        self.active = []
        self.peak_motion = 0.0
        self.departure_streak = 0
        self.return_since = None
        self.recent.clear()

    def _stable(self) -> bool:
        if len(self.recent) < 4 or (
                self.recent[-1].timestamp-self.recent[0].timestamp < self.config.settle_seconds):
            return False
        centers = np.array([sample.center for sample in self.recent])
        scales = np.array([sample.scale for sample in self.recent])
        if np.max(np.linalg.norm(centers-centers.mean(axis=0), axis=1)) > (
                self.config.stable_center_torso_lengths*float(np.median(scales))):
            return False
        if (float(scales.max()-scales.min())/float(np.median(scales)) >
                self.config.stable_scale_fraction):
            return False
        common = [joint for joint in MOTION_JOINTS
                  if all(joint in sample.points for sample in self.recent)]
        if len(common) < 2:
            return False
        for joint in common:
            points = np.array([sample.points[joint] for sample in self.recent])
            if np.max(np.linalg.norm(points-points.mean(axis=0), axis=1)) > (
                    self.config.stable_joint_torso_lengths):
                return False
        return True

    def _in_zone(self, sample: PoseSample) -> bool:
        return (self.baseline_center is not None and
                math.dist(sample.center, self.baseline_center) <=
                self.config.zone_center_torso_lengths*self.baseline_scale and
                abs(sample.scale-self.baseline_scale)/self.baseline_scale <=
                self.config.zone_scale_fraction)

    def _motion(self, sample: PoseSample) -> float:
        distances = sorted((math.dist(sample.points[joint], base)
                            for joint, base in self.baseline.items()
                            if joint in sample.points), reverse=True)
        return sum(distances[:2])/2 if len(distances) >= 2 else 0.0

    def update(self, sample: PoseSample | None, timestamp: float) -> None:
        if self.started_at is None:
            self.started_at = timestamp
        if sample is None:
            if (self.last_valid_time is not None and timestamp-self.last_valid_time >
                    self.config.max_missing_seconds):
                self._settle(timestamp, "tracking lost")
            return
        if (self.last_valid_time is not None and timestamp-self.last_valid_time >
                self.config.max_missing_seconds):
            self._settle(timestamp, "tracking gap")
        self.last_valid_time = timestamp
        self.recent.append(sample)
        while (self.recent and timestamp-self.recent[0].timestamp >
               self.config.settle_seconds+0.12):
            self.recent.popleft()

        if self.state == "SETTLING":
            if (timestamp-self.started_at >= self.config.min_walk_away_seconds and
                    self._stable()):
                self.baseline = {
                    joint: tuple(np.mean([s.points[joint] for s in self.recent], axis=0))
                    for joint in MOTION_JOINTS
                    if all(joint in s.points for s in self.recent)
                }
                self.baseline_center = tuple(np.mean([s.center for s in self.recent], axis=0))
                self.baseline_scale = float(np.median([s.scale for s in self.recent]))
                self.state = "READY"
                self.events.append({"time_seconds": round(timestamp, 3), "event": "armed"})
            return

        if not self._in_zone(sample):
            self._settle(timestamp, "person moved from exercise spot")
            return
        motion = self._motion(sample)
        if self.state == "READY":
            self.departure_streak = (self.departure_streak+1 if motion >=
                                     self.config.start_motion_torso_lengths else 0)
            if self.departure_streak >= 2:
                self.state = "MOVING"
                self.active = [s for s in self.recent if timestamp-s.timestamp <= 0.4]
                self.peak_motion = motion
                self.events.append({"time_seconds": round(timestamp, 3),
                                    "event": "candidate_started"})
            return

        self.active.append(sample)
        self.peak_motion = max(self.peak_motion, motion)
        duration = timestamp-self.active[0].timestamp
        if duration > self.config.max_rep_seconds:
            self._settle(timestamp, "movement timed out")
            return
        if motion <= self.config.return_motion_torso_lengths:
            self.return_since = timestamp if self.return_since is None else self.return_since
        else:
            self.return_since = None
        if (self.return_since is not None and
                timestamp-self.return_since >= self.config.return_hold_seconds and
                duration >= self.config.min_rep_seconds and
                self.peak_motion >= self.config.min_peak_motion_torso_lengths):
            self.candidates.append(self.active)
            self.events.append({"time_seconds": round(timestamp, 3),
                                "event": "candidate_completed",
                                "duration_seconds": round(duration, 3)})
            self.state = "READY"
            self.active = []
            self.departure_streak = 0
            self.return_since = None
            self.peak_motion = 0.0
