"""Front-view, one-arm shoulder abduction measured in the 2D image plane."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .landmarks import Landmarks


class AbductionState(str, Enum):
    READY = "READY"
    RAISING = "RAISING"
    TOP = "TOP"
    LOWERING = "LOWERING"


@dataclass(frozen=True)
class AbductionConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_landmark_confidence: float = 0.65
    frame_margin: float = 0.025
    minimum_shoulder_width_to_torso: float = 0.55
    smoothing_tau_seconds: float = 0.12
    max_missing_seconds: float = 0.75
    settle_seconds: float = 0.55
    stable_center_torso_lengths: float = 0.18
    stable_scale_fraction: float = 0.09
    motion_center_torso_lengths: float = 0.38
    motion_scale_fraction: float = 0.20
    rest_angle_degrees: float = 22.0
    start_angle_degrees: float = 34.0
    minimum_cycle_angle_degrees: float = 55.0
    target_angle_degrees: float = 165.0
    target_hold_seconds: float = 0.15
    lower_from_peak_degrees: float = 12.0
    return_angle_degrees: float = 25.0
    minimum_elbow_angle_degrees: float = 150.0
    maximum_torso_lean_degrees: float = 18.0
    minimum_cycle_seconds: float = 0.8
    maximum_cycle_seconds: float = 35.0


@dataclass(frozen=True)
class AbductionMetrics:
    shoulder_angle: float
    hand_angle: float
    elbow_angle: float
    torso_lean: float
    outward_reach: float


@dataclass(frozen=True)
class AbductionFrameResult:
    state: str
    rep_count: int
    full_arc_count: int
    side: str | None
    reliability: float
    metrics: AbductionMetrics | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    if denominator < 1e-6:
        raise ValueError("Coincident landmarks cannot define an angle")
    cosine = float(np.clip(np.dot(a, b)/denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class ShoulderAbductionAnalyzer:
    """Counts raise-and-return cycles; flags whether a near-180° 2D arc was seen."""

    def __init__(self, config: AbductionConfig) -> None:
        self.config = config
        self.state = AbductionState.READY
        self.side: str | None = None
        self.smoothed: AbductionMetrics | None = None
        self.last_valid_time: float | None = None
        self.rest_samples: deque[tuple[float, tuple[float, float], float]] = deque()
        self.armed = False
        self.anchor_center: tuple[float, float] | None = None
        self.anchor_scale: float | None = None
        self.attempt: dict | None = None
        self.reps: list[dict] = []
        self.full_arc_count = 0
        self.completion_message: str | None = None
        self.completion_until = -1.0

    def _reset_movement(self) -> None:
        self.state = AbductionState.READY
        self.side = None
        self.smoothed = None
        self.rest_samples.clear()
        self.armed = False
        self.anchor_center = None
        self.anchor_scale = None
        self.attempt = None

    def _result(self, reliability: float, feedback: str, valid: bool = False,
                warning: bool = False) -> AbductionFrameResult:
        return AbductionFrameResult(self.state.value, len(self.reps),
                                    self.full_arc_count, self.side, reliability,
                                    self.smoothed if valid else None, feedback,
                                    valid, warning)

    def _missing(self, timestamp: float, reliability: float,
                 feedback: str = "Unable to reliably assess") -> AbductionFrameResult:
        if (self.last_valid_time is not None and
                timestamp-self.last_valid_time > self.config.max_missing_seconds):
            self._reset_movement()
        return self._result(reliability, feedback)

    def _steady_start(self, timestamp: float, center: tuple[float, float],
                      scale: float) -> bool:
        c = self.config
        self.rest_samples.append((timestamp, center, scale))
        while self.rest_samples and timestamp-self.rest_samples[0][0] > c.settle_seconds+0.12:
            self.rest_samples.popleft()
        if (len(self.rest_samples) < 4 or
                timestamp-self.rest_samples[0][0] < c.settle_seconds):
            return False
        centers = np.array([item[1] for item in self.rest_samples])
        scales = np.array([item[2] for item in self.rest_samples])
        median_scale = float(np.median(scales))
        return (float(np.max(np.linalg.norm(centers-centers.mean(axis=0), axis=1))) <=
                c.stable_center_torso_lengths*median_scale and
                float(scales.max()-scales.min())/median_scale <= c.stable_scale_fraction)

    def _in_exercise_spot(self, center: tuple[float, float], scale: float) -> bool:
        if self.anchor_center is None or self.anchor_scale is None:
            return False
        return (math.dist(center, self.anchor_center) <=
                self.config.motion_center_torso_lengths*self.anchor_scale and
                abs(scale-self.anchor_scale)/self.anchor_scale <=
                self.config.motion_scale_fraction)

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> AbductionFrameResult:
        c = self.config
        core_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        core = [landmarks.get(name) for name in core_names]
        if any(point is None or point.confidence < c.min_landmark_confidence
               for point in core):
            return self._missing(timestamp, 0.0)
        core_reliability = min(point.confidence for point in core)
        if any(not (c.frame_margin <= point.x <= 1-c.frame_margin and
                    c.frame_margin <= point.y <= 1-c.frame_margin) for point in core):
            return self._missing(timestamp, core_reliability,
                                 "Move back so shoulders and hips are visible")
        def xy(point: object) -> np.ndarray:
            return np.array((point.x*width, point.y*height), dtype=float)
        shoulder_mid = (xy(core[0])+xy(core[1]))/2
        hip_mid = (xy(core[2])+xy(core[3]))/2
        torso = hip_mid-shoulder_mid
        torso_length = float(np.linalg.norm(torso))
        if torso_length < 25:
            return self._missing(timestamp, core_reliability)
        shoulder_width = float(np.linalg.norm(xy(core[0])-xy(core[1])))
        if shoulder_width/torso_length < c.minimum_shoulder_width_to_torso:
            return self._missing(timestamp, core_reliability, "Face the camera")
        torso_lean = math.degrees(math.atan2(abs(torso[0]), abs(torso[1])))
        candidates = []
        sides = (self.side,) if self.state != AbductionState.READY and self.side else (
            "left", "right")
        out_of_frame = False
        for side in sides:
            shoulder = landmarks[f"{side}_shoulder"]
            other = landmarks[f"{'right' if side == 'left' else 'left'}_shoulder"]
            elbow = landmarks.get(f"{side}_elbow")
            wrist = landmarks.get(f"{side}_wrist")
            if (elbow is None or wrist is None or
                    min(elbow.confidence, wrist.confidence) < c.min_landmark_confidence):
                continue
            if any(not (c.frame_margin <= point.x <= 1-c.frame_margin and
                        c.frame_margin <= point.y <= 1-c.frame_margin)
                   for point in (elbow, wrist)):
                out_of_frame = True
                continue
            shoulder_xy, elbow_xy, wrist_xy = xy(shoulder), xy(elbow), xy(wrist)
            try:
                shoulder_angle = _angle(torso, elbow_xy-shoulder_xy)
                hand_angle = _angle(torso, wrist_xy-shoulder_xy)
                elbow_angle = _angle(shoulder_xy-elbow_xy, wrist_xy-elbow_xy)
            except ValueError:
                continue
            outward_sign = 1 if shoulder.x > other.x else -1
            outward = ((wrist_xy[0]-shoulder_xy[0])*outward_sign/torso_length)
            reliability = min(core_reliability, elbow.confidence, wrist.confidence)
            candidates.append((side, reliability, AbductionMetrics(
                shoulder_angle, hand_angle, elbow_angle, torso_lean, outward)))
        if not candidates:
            return self._missing(timestamp, core_reliability,
                                 "Move back so your whole arm is visible" if out_of_frame else
                                 "Unable to reliably assess")
        side, reliability, raw = max(candidates,
                                     key=lambda item: (item[2].shoulder_angle,
                                                       item[1]))
        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > c.max_missing_seconds:
            self._reset_movement()
        if self.side != side:
            self.smoothed = None
        self.side = side
        alpha = (1.0 if self.smoothed is None or gap is None or gap <= 0 else
                 1-math.exp(-gap/c.smoothing_tau_seconds))
        if self.smoothed is None:
            self.smoothed = raw
        else:
            old = self.smoothed
            self.smoothed = AbductionMetrics(*(
                getattr(old, field)+alpha*(getattr(raw, field)-getattr(old, field))
                for field in ("shoulder_angle", "hand_angle", "elbow_angle",
                              "torso_lean", "outward_reach")))
        self.last_valid_time = timestamp
        m = self.smoothed
        center = (float(hip_mid[0]), float(hip_mid[1]))
        if self.state != AbductionState.READY and not self._in_exercise_spot(
                center, torso_length):
            self._reset_movement()
            return self._result(reliability, "Walk/camera movement excluded")

        warning = False
        if self.state == AbductionState.READY:
            if (m.shoulder_angle <= c.rest_angle_degrees and
                    m.torso_lean <= c.maximum_torso_lean_degrees):
                if self._steady_start(timestamp, center, torso_length):
                    self.armed = True
                    self.anchor_center = tuple(np.median(
                        [item[1] for item in self.rest_samples], axis=0))
                    self.anchor_scale = float(np.median(
                        [item[2] for item in self.rest_samples]))
            else:
                self.rest_samples.clear()
            if self.armed and not self._in_exercise_spot(center, torso_length):
                self._reset_movement()
                return self._result(reliability, "Walk/camera movement excluded")
            if self.armed and m.shoulder_angle >= c.start_angle_degrees:
                self.state = AbductionState.RAISING
                self.attempt = {"start": timestamp, "side": side,
                                "peak": m.shoulder_angle,
                                "minimum_elbow": 180.0,
                                "maximum_torso": m.torso_lean,
                                "maximum_midrange_outward": 0.0,
                                "minimum_reliability": reliability,
                                "top_since": None, "reached_top": False}
                self.rest_samples.clear()
                self.armed = False
        else:
            a = self.attempt
            a["peak"] = max(a["peak"], m.shoulder_angle)
            a["maximum_torso"] = max(a["maximum_torso"], m.torso_lean)
            a["minimum_reliability"] = min(a["minimum_reliability"], reliability)
            if m.shoulder_angle >= c.minimum_cycle_angle_degrees:
                a["minimum_elbow"] = min(a["minimum_elbow"], m.elbow_angle)
            if 60 <= m.shoulder_angle <= 125:
                a["maximum_midrange_outward"] = max(
                    a["maximum_midrange_outward"], m.outward_reach)
            if timestamp-a["start"] > c.maximum_cycle_seconds:
                self._reset_movement()
                return self._result(reliability, "Movement timed out; start again")
            # The movement target is the upper-arm angle. A wrist-position gate
            # makes the cue oscillate when the elbow bends or tracking shifts.
            at_top = m.shoulder_angle >= c.target_angle_degrees
            if at_top:
                a["top_since"] = (timestamp if a["top_since"] is None else
                                   a["top_since"])
                if timestamp-a["top_since"] >= c.target_hold_seconds:
                    a["reached_top"] = True
                    self.state = AbductionState.TOP
            else:
                a["top_since"] = None
            if self.state == AbductionState.RAISING:
                if (not a["reached_top"] and
                        a["peak"]-m.shoulder_angle >= c.lower_from_peak_degrees and
                        a["peak"] >= c.minimum_cycle_angle_degrees):
                    self.state = AbductionState.LOWERING
            elif self.state == AbductionState.TOP:
                if (a["peak"]-m.shoulder_angle >= c.lower_from_peak_degrees):
                    self.state = AbductionState.LOWERING
            if (self.state == AbductionState.LOWERING and
                    m.shoulder_angle <= c.return_angle_degrees and
                    timestamp-a["start"] >= c.minimum_cycle_seconds):
                full = bool(a["reached_top"])
                if full:
                    self.full_arc_count += 1
                self.reps.append({
                    "number": len(self.reps)+1,
                    "side": a["side"],
                    "duration_seconds": round(timestamp-a["start"], 3),
                    "maximum_shoulder_abduction_degrees_2d": round(a["peak"], 2),
                    "minimum_elbow_angle_degrees_2d": round(a["minimum_elbow"], 2),
                    "maximum_torso_lean_degrees_2d": round(a["maximum_torso"], 2),
                    "maximum_midrange_outward_torso_lengths": float(round(
                        a["maximum_midrange_outward"], 3)),
                    "full_arc_reached": full,
                    "elbow_bend_flag": (a["minimum_elbow"] <
                                        c.minimum_elbow_angle_degrees),
                    "torso_lean_flag": (a["maximum_torso"] >
                                        c.maximum_torso_lean_degrees),
                    "minimum_landmark_reliability": round(a["minimum_reliability"], 3),
                })
                self.completion_message = ("Full overhead arc recorded" if full else
                                           "Short arc: raise higher next time")
                self.completion_until = timestamp+2.0
                self._reset_movement()
                return self._result(reliability, self.completion_message, True,
                                    not full)

        if self.state == AbductionState.READY:
            feedback = (self.completion_message if timestamp < self.completion_until else
                        "Hold arm down and stand still to begin" if not self.armed else
                        "Raise one arm out to the side")
            warning = ((timestamp < self.completion_until and bool(
                self.completion_message and self.completion_message.startswith("Short"))) or
                (not self.armed and m.shoulder_angle > c.rest_angle_degrees))
        elif m.torso_lean > c.maximum_torso_lean_degrees:
            feedback, warning = "Keep your torso upright", True
        elif (m.shoulder_angle >= c.minimum_cycle_angle_degrees and
              m.elbow_angle < c.minimum_elbow_angle_degrees):
            feedback, warning = "Straighten your elbow", True
        elif self.state == AbductionState.TOP:
            feedback = "Overhead height reached; lower slowly"
        elif self.state == AbductionState.LOWERING:
            feedback = ("Lower slowly to your side" if a["reached_top"] else
                        "Lower your arm to your side to finish")
            warning = not a["reached_top"]
        else:
            feedback = "Raise your arm out and up"
        return self._result(reliability, feedback, True, warning)
