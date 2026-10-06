"""Front-view, one-arm cross-body stretch over generic normalized landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .landmarks import Landmarks


class ArmState(str, Enum):
    READY = "READY"
    REACHING = "REACHING"
    HOLDING = "HOLDING"
    RETURNING = "RETURNING"


@dataclass(frozen=True)
class ArmStretchConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_landmark_confidence: float = 0.65
    frame_margin: float = 0.025
    minimum_shoulder_width: float = 0.16
    smoothing_tau_seconds: float = 0.12
    max_missing_seconds: float = 0.75
    start_reach: float = 0.30
    target_reach: float = 0.75
    release_reach: float = 0.55
    rest_reach: float = 0.15
    return_complete_reach: float = 0.30
    maximum_wrist_height_offset: float = 0.45
    hold_seconds: float = 1.5
    max_attempt_seconds: float = 15.0


@dataclass
class ArmMetrics:
    reach: float
    wrist_height_offset: float
    hold_seconds: float


@dataclass
class ArmFrameResult:
    state: str
    rep_count: int
    side: str | None
    reliability: float
    metrics: ArmMetrics | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False


class CrossBodyStretchAnalyzer:
    def __init__(self, config: ArmStretchConfig) -> None:
        self.config = config
        self.state = ArmState.READY
        self.side: str | None = None
        self.ready = False
        self.smoothed: ArmMetrics | None = None
        self.last_valid_time: float | None = None
        self.attempt: dict | None = None
        self.reps: list[dict] = []

    def _reset(self) -> None:
        self.state = ArmState.READY
        self.side = None
        self.ready = False
        self.smoothed = None
        self.attempt = None

    def _result(self, reliability: float, feedback: str, valid: bool = False,
                warning: bool = False) -> ArmFrameResult:
        return ArmFrameResult(self.state.value, len(self.reps), self.side, reliability,
                              self.smoothed if valid else None, feedback, valid, warning)

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> ArmFrameResult:
        c = self.config
        shoulders = (landmarks.get("left_shoulder"), landmarks.get("right_shoulder"))
        if any(p is None or p.confidence < c.min_landmark_confidence for p in shoulders):
            return self._missing(timestamp, 0.0)
        left, right = shoulders
        shoulder_width = abs(left.x-right.x)
        if shoulder_width < c.minimum_shoulder_width:
            return self._result(min(left.confidence, right.confidence), "Face the camera")

        candidates = []
        for side, own, other in (("left", left, right), ("right", right, left)):
            wrist = landmarks.get(f"{side}_wrist")
            if wrist is None:
                continue
            reliability = min(own.confidence, other.confidence, wrist.confidence)
            if reliability < c.min_landmark_confidence:
                continue
            points = (own, other, wrist)
            if any(not (c.frame_margin <= p.x <= 1-c.frame_margin and
                        c.frame_margin <= p.y <= 1-c.frame_margin) for p in points):
                candidates.append((side, reliability, None))
                continue
            reach = (wrist.x-own.x)/(other.x-own.x)
            height_offset = abs(wrist.y-(own.y+other.y)/2)*height/(shoulder_width*width)
            candidates.append((side, reliability, (reach, height_offset)))

        if self.state != ArmState.READY and self.side:
            candidates = [item for item in candidates if item[0] == self.side]
        if not candidates:
            return self._missing(timestamp, 0.0)
        usable = [item for item in candidates if item[2] is not None]
        if not usable:
            return self._missing(timestamp, max(item[1] for item in candidates),
                                 "Move back so your stretching wrist is visible")
        side, reliability, measurements = max(usable, key=lambda item: item[2][0])
        if self.side != side:
            self.smoothed = None
        self.side = side

        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > c.max_missing_seconds:
            self._reset()
            self.side = side
        alpha = (1.0 if self.smoothed is None or gap is None or gap <= 0 else
                 1-math.exp(-gap/c.smoothing_tau_seconds))
        reach, height_offset = measurements
        if self.smoothed is not None:
            reach = self.smoothed.reach + alpha*(reach-self.smoothed.reach)
            height_offset = self.smoothed.wrist_height_offset + alpha*(height_offset-self.smoothed.wrist_height_offset)
        self.last_valid_time = timestamp
        hold_elapsed = 0.0
        feedback_override = None

        if self.state == ArmState.READY:
            if reach <= c.rest_reach:
                self.ready = True
            if self.ready and reach >= c.start_reach:
                self.state = ArmState.REACHING
                self.attempt = {"start_time": timestamp, "max_reach": reach,
                                "min_reliability": reliability, "hold_start": None,
                                "max_hold_seconds": 0.0}
        elif self.attempt is not None:
            a = self.attempt
            a["max_reach"] = max(a["max_reach"], reach)
            a["min_reliability"] = min(a["min_reliability"], reliability)
            if timestamp-a["start_time"] > c.max_attempt_seconds:
                self._reset()
                feedback_override = "Start again"
            elif self.state == ArmState.REACHING:
                if reach <= c.rest_reach:
                    self._reset()
                    self.ready = True
                elif (reach >= c.target_reach and
                      height_offset <= c.maximum_wrist_height_offset):
                    self.state = ArmState.HOLDING
                    a["hold_start"] = timestamp
            elif self.state == ArmState.HOLDING:
                good_position = (reach >= c.target_reach and
                                 height_offset <= c.maximum_wrist_height_offset)
                if good_position:
                    hold_elapsed = timestamp-a["hold_start"]
                    a["max_hold_seconds"] = max(a["max_hold_seconds"], hold_elapsed)
                elif reach < c.release_reach:
                    if a["max_hold_seconds"] >= c.hold_seconds:
                        self.state = ArmState.RETURNING
                    else:
                        self._reset()
                        feedback_override = "Hold longer before returning"
                elif a["max_hold_seconds"] < c.hold_seconds:
                    self.state = ArmState.REACHING
                    a["hold_start"] = None
                    if reach < c.target_reach:
                        feedback_override = "Hold longer before returning"
            elif self.state == ArmState.RETURNING:
                if reach <= c.return_complete_reach:
                    self.reps.append({
                        "number": len(self.reps)+1,
                        "side": self.side,
                        "duration_seconds": round(timestamp-a["start_time"], 3),
                        "hold_seconds": round(a["max_hold_seconds"], 3),
                        "maximum_cross_body_reach": round(a["max_reach"], 3),
                        "minimum_landmark_reliability": round(a["min_reliability"], 3),
                        "arm_straightness": "not_assessed",
                    })
                    self.state = ArmState.READY
                    self.attempt = None
                    self.ready = True

        warning = (self.state in (ArmState.REACHING, ArmState.HOLDING) and
                   reach >= c.target_reach and
                   height_offset > c.maximum_wrist_height_offset)
        if feedback_override:
            feedback = feedback_override
            warning = True
        elif warning:
            feedback = "Keep your arm near shoulder height"
        elif self.state == ArmState.HOLDING:
            feedback = ("Hold complete - return your arm" if self.attempt and
                        self.attempt["max_hold_seconds"] >= c.hold_seconds else "Hold the stretch")
        elif self.state == ArmState.RETURNING:
            feedback = "Return your arm to start"
        elif self.state == ArmState.REACHING:
            feedback = "Reach across your chest"
        else:
            feedback = "Ready for cross-body stretch" if self.ready else "Lower one arm to start"
        hold_display = self.attempt["max_hold_seconds"] if self.attempt else hold_elapsed
        self.smoothed = ArmMetrics(reach, height_offset, hold_display)
        return self._result(reliability, feedback, True, warning)

    def _missing(self, timestamp: float, reliability: float,
                 feedback: str = "Unable to reliably assess") -> ArmFrameResult:
        if self.last_valid_time is not None and timestamp-self.last_valid_time > self.config.max_missing_seconds:
            self._reset()
        return self._result(reliability, feedback)
