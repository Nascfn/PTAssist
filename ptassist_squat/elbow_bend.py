"""One-arm bend-and-straighten demo using generic 2D landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .landmarks import Landmarks


class ElbowState(str, Enum):
    STRAIGHT = "STRAIGHT"
    BENDING = "BENDING"
    BENT = "BENT"
    STRAIGHTENING = "STRAIGHTENING"


@dataclass(frozen=True)
class ElbowConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = .5
    min_landmark_confidence: float = .65
    frame_margin: float = .025
    smoothing_tau_seconds: float = .12
    max_missing_seconds: float = .75
    settle_seconds: float = .45
    rest_angle_degrees: float = 155
    start_angle_degrees: float = 140
    bend_target_degrees: float = 95
    reversal_degrees: float = 14
    return_angle_degrees: float = 155
    upper_arm_start_limit_degrees: float = 40
    upper_arm_warning_degrees: float = 50
    shoulder_motion_upper_arm_lengths: float = .9
    scale_change_fraction: float = .3
    minimum_cycle_seconds: float = .6
    maximum_cycle_seconds: float = 20


@dataclass(frozen=True)
class ElbowMetrics:
    elbow_angle: float
    upper_arm_angle: float


@dataclass(frozen=True)
class ElbowFrameResult:
    state: str
    rep_count: int
    full_bend_count: int
    side: str | None
    reliability: float
    metrics: ElbowMetrics | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    length = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if length < 1e-6:
        raise ValueError("Coincident arm landmarks")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2)/length, -1, 1))))


class ElbowBendAnalyzer:
    """Counts a visible bend-and-return; the target is a POC cutoff."""

    def __init__(self, config: ElbowConfig) -> None:
        self.config = config
        self.state = ElbowState.STRAIGHT
        self.side: str | None = None
        self.smoothed: ElbowMetrics | None = None
        self.last_valid_time: float | None = None
        self.rest_since: float | None = None
        self.armed = False
        self.anchor: np.ndarray | None = None
        self.anchor_scale: float | None = None
        self.attempt: dict | None = None
        self.reps: list[dict] = []
        self.full_bend_count = 0
        self.completion_message = ""
        self.completion_until = -1.0

    def _reset(self) -> None:
        self.state = ElbowState.STRAIGHT
        self.side = None
        self.smoothed = None
        self.rest_since = None
        self.armed = False
        self.anchor = None
        self.anchor_scale = None
        self.attempt = None

    def _result(self, confidence: float, feedback: str, valid: bool = False,
                warning: bool = False) -> ElbowFrameResult:
        return ElbowFrameResult(self.state.value, len(self.reps),
                                self.full_bend_count, self.side, confidence,
                                self.smoothed if valid else None, feedback,
                                valid, warning)

    def _missing(self, now: float, confidence: float,
                 feedback: str = "Unable to reliably assess") -> ElbowFrameResult:
        if (self.last_valid_time is not None and
                now-self.last_valid_time > self.config.max_missing_seconds):
            self._reset()
        return self._result(confidence, feedback)

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> ElbowFrameResult:
        c = self.config
        candidates = []
        out_of_frame = False
        for side in ((self.side,) if self.state != ElbowState.STRAIGHT and self.side
                     else ("left", "right")):
            points = [landmarks.get(f"{side}_{name}") for name in
                      ("shoulder", "elbow", "wrist")]
            if any(p is None or p.confidence < c.min_landmark_confidence for p in points):
                continue
            if any(not (c.frame_margin <= p.x <= 1-c.frame_margin and
                        c.frame_margin <= p.y <= 1-c.frame_margin) for p in points):
                out_of_frame = True
                continue
            shoulder, elbow, wrist = [np.array((p.x*width, p.y*height))
                                      for p in points]
            upper, forearm = shoulder-elbow, wrist-elbow
            upper_length = float(np.linalg.norm(upper))
            if upper_length < 25 or np.linalg.norm(forearm) < 25:
                continue
            try:
                metrics = ElbowMetrics(_angle(upper, forearm),
                                       _angle(elbow-shoulder, np.array((0., 1.))))
            except ValueError:
                continue
            candidates.append((side, min(p.confidence for p in points), metrics,
                               shoulder, upper_length))
        if not candidates:
            return self._missing(timestamp, 0.0,
                                 "Keep shoulder, elbow, and hand in view" if out_of_frame
                                 else "Unable to reliably assess")
        # Before a rep, prefer the arm showing the most bend; lock it after start.
        side, confidence, raw, shoulder, scale = min(
            candidates, key=lambda x: (x[2].elbow_angle, -x[1]))
        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > c.max_missing_seconds:
            self._reset()
        if self.side != side:
            self.smoothed = None
        self.side = side
        alpha = (1.0 if self.smoothed is None or gap is None or gap <= 0 else
                 1-math.exp(-gap/c.smoothing_tau_seconds))
        if self.smoothed is None:
            self.smoothed = raw
        else:
            old = self.smoothed
            self.smoothed = ElbowMetrics(
                old.elbow_angle+alpha*(raw.elbow_angle-old.elbow_angle),
                old.upper_arm_angle+alpha*(raw.upper_arm_angle-old.upper_arm_angle))
        self.last_valid_time = timestamp
        m = self.smoothed
        if (self.anchor is not None and
                (np.linalg.norm(shoulder-self.anchor) >
                 c.shoulder_motion_upper_arm_lengths*self.anchor_scale or
                 abs(scale-self.anchor_scale)/self.anchor_scale >
                 c.scale_change_fraction)):
            self._reset()
            return self._result(confidence, "Moved out of position; start again")

        if self.state == ElbowState.STRAIGHT:
            at_rest = (m.elbow_angle >= c.rest_angle_degrees and
                       m.upper_arm_angle <= c.upper_arm_start_limit_degrees)
            if at_rest:
                if self.rest_since is None:
                    self.rest_since = timestamp
                if timestamp-self.rest_since >= c.settle_seconds:
                    self.armed = True
            else:
                self.rest_since = None
            if self.armed and m.elbow_angle <= c.start_angle_degrees:
                self.state = ElbowState.BENDING
                self.anchor, self.anchor_scale = shoulder, scale
                self.attempt = {"start": timestamp, "side": side,
                                "minimum_angle": m.elbow_angle,
                                "maximum_upper_arm_angle": m.upper_arm_angle,
                                "minimum_reliability": confidence,
                                "reached_bend": False}
                self.armed = False
        else:
            a = self.attempt
            a["minimum_angle"] = min(a["minimum_angle"], m.elbow_angle)
            a["maximum_upper_arm_angle"] = max(
                a["maximum_upper_arm_angle"], m.upper_arm_angle)
            a["minimum_reliability"] = min(a["minimum_reliability"], confidence)
            if m.elbow_angle <= c.bend_target_degrees:
                a["reached_bend"] = True
                self.state = ElbowState.BENT
            if (self.state in (ElbowState.BENDING, ElbowState.BENT) and
                    m.elbow_angle-a["minimum_angle"] >= c.reversal_degrees):
                self.state = ElbowState.STRAIGHTENING
            if timestamp-a["start"] > c.maximum_cycle_seconds:
                self._reset()
                return self._result(confidence, "Movement timed out; start again")
            if (self.state == ElbowState.STRAIGHTENING and
                    m.elbow_angle >= c.return_angle_degrees and
                    timestamp-a["start"] >= c.minimum_cycle_seconds):
                full = bool(a["reached_bend"])
                self.full_bend_count += int(full)
                self.reps.append({
                    "number": len(self.reps)+1, "side": a["side"],
                    "duration_seconds": round(timestamp-a["start"], 3),
                    "minimum_elbow_angle_degrees_2d": round(a["minimum_angle"], 2),
                    "maximum_upper_arm_angle_degrees_2d": round(
                        a["maximum_upper_arm_angle"], 2),
                    "full_bend_reached": full,
                    "upper_arm_moved_flag": (a["maximum_upper_arm_angle"] >
                                             c.upper_arm_warning_degrees),
                    "minimum_landmark_reliability": round(
                        a["minimum_reliability"], 3),
                })
                self.completion_message = ("Bend and return recorded" if full else
                                           "Short bend recorded")
                self.completion_until = timestamp+2
                self._reset()
                return self._result(confidence, self.completion_message, True, not full)

        if self.state == ElbowState.STRAIGHT:
            if timestamp < self.completion_until:
                feedback = self.completion_message
                warning = feedback.startswith("Short")
            elif m.upper_arm_angle > c.upper_arm_start_limit_degrees:
                feedback, warning = "Lower your upper arm to your side", True
            elif self.armed:
                feedback, warning = "Bend one elbow toward your shoulder", False
            else:
                feedback, warning = "Straighten one arm at your side to start", True
        elif m.upper_arm_angle > c.upper_arm_warning_degrees:
            feedback, warning = "Keep your upper arm near your side", True
        elif self.state == ElbowState.BENT:
            feedback, warning = "Bend reached; straighten your elbow", False
        elif self.state == ElbowState.STRAIGHTENING:
            feedback, warning = "Straighten your elbow to finish", False
        else:
            feedback, warning = "Bend your elbow", False
        return self._result(confidence, feedback, True, warning)
