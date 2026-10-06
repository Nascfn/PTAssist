"""Squat logic over generic landmarks; no MediaPipe or camera dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .geometry import joint_angle, lean_from_vertical
from .landmarks import Landmarks


REQUIRED = ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")


class State(str, Enum):
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


@dataclass(frozen=True)
class Config:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_landmark_confidence: float = 0.65
    frame_margin: float = 0.025
    max_side_width_to_torso: float = 0.75
    min_opposite_ankle_confidence: float = 0.65
    min_opposite_foot_confidence: float = 0.6
    max_ankle_separation_to_torso: float = 0.6
    side_switch_advantage: float = 0.12
    smoothing_tau_seconds: float = 0.12
    max_missing_seconds: float = 0.75
    standing_knee_angle: float = 165.0
    start_descending_knee_angle: float = 155.0
    bottom_entry_knee_angle: float = 145.0
    ascent_from_bottom_degrees: float = 6.0
    depth_knee_angle: float = 110.0
    excessive_torso_lean_degrees: float = 45.0
    max_rep_duration_seconds: float = 12.0
    reference_knee_tolerance_degrees: float = 12.0
    reference_torso_tolerance_degrees: float = 10.0
    reference_duration_tolerance_fraction: float = 0.4


@dataclass
class Angles:
    knee: float
    hip: float
    torso: float


@dataclass
class FrameResult:
    state: str
    rep_count: int
    side: str | None
    reliability: float
    angles: Angles | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False
    stance_ratio: float | None = None
    stance_source: str | None = None
    stance_reliability: float | None = None


class SquatAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = State.STANDING
        self.ready = False
        self.side: str | None = None
        self.smoothed: Angles | None = None
        self.last_valid_time: float | None = None
        self.attempt: dict | None = None
        self.reps: list[dict] = []

    def _clear_attempt(self) -> None:
        self.state = State.STANDING
        self.ready = False
        self.attempt = None
        self.smoothed = None

    def _side_score(self, landmarks: Landmarks, side: str) -> float:
        points = [landmarks.get(f"{side}_{part}") for part in REQUIRED]
        if any(p is None for p in points):
            return 0.0
        return min(p.confidence for p in points) * 0.7 + sum(p.confidence for p in points) / len(points) * 0.3

    def _choose_side(self, landmarks: Landmarks) -> tuple[str | None, float]:
        scores = {side: self._side_score(landmarks, side) for side in ("left", "right")}
        candidate = max(scores, key=scores.get)
        if self.side and scores[self.side] + self.config.side_switch_advantage >= scores[candidate]:
            candidate = self.side
        self.side = candidate
        points = [landmarks.get(f"{candidate}_{part}") for part in REQUIRED]
        reliability = min(p.confidence for p in points) if all(points) else 0.0
        return candidate, reliability

    def _result(self, side: str | None, reliability: float, feedback: str,
                valid: bool = False, form_warning: bool = False,
                stance_ratio: float | None = None,
                stance_source: str | None = None,
                stance_reliability: float | None = None) -> FrameResult:
        return FrameResult(self.state.value, len(self.reps), side, reliability,
                           self.smoothed if valid else None, feedback, valid, form_warning,
                           stance_ratio, stance_source, stance_reliability)

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> FrameResult:
        config = self.config
        side, reliability = self._choose_side(landmarks) if landmarks else (None, 0.0)
        if side is None or reliability < config.min_landmark_confidence:
            if self.last_valid_time is not None and timestamp - self.last_valid_time > config.max_missing_seconds:
                self._clear_attempt()
            return self._result(side, reliability, "Unable to reliably assess")

        nose = landmarks.get("nose")
        if nose is None or nose.confidence < config.min_landmark_confidence:
            return self._result(side, min(reliability, nose.confidence if nose else 0.0),
                                "Unable to reliably assess")
        reliability = min(reliability, nose.confidence)
        selected = [landmarks[f"{side}_{part}"] for part in REQUIRED]
        if any(not (config.frame_margin <= p.x <= 1-config.frame_margin and
                    config.frame_margin <= p.y <= 1-config.frame_margin) for p in selected):
            if self.last_valid_time is not None and timestamp - self.last_valid_time > config.max_missing_seconds:
                self._clear_attempt()
            return self._result(side, reliability, "Move back so your full body is visible")
        if not (config.frame_margin <= nose.x <= 1-config.frame_margin and
                config.frame_margin <= nose.y <= 1-config.frame_margin):
            return self._result(side, reliability, "Move back so your full body is visible")

        shoulder, hip, knee, ankle, heel, foot = selected
        torso_length = math.hypot((shoulder.x-hip.x)*width, (shoulder.y-hip.y)*height)
        if torso_length < 15:
            return self._result(side, reliability, "Unable to reliably assess")
        opposite_side = "right" if side == "left" else "left"
        other_ankle = landmarks.get(f"{opposite_side}_ankle")
        other_heel = landmarks.get(f"{opposite_side}_heel")
        other_foot = landmarks.get(f"{opposite_side}_foot_index")
        in_frame = lambda p: (config.frame_margin <= p.x <= 1-config.frame_margin and
                              config.frame_margin <= p.y <= 1-config.frame_margin)
        ankle_reliable = (other_ankle is not None and
                          other_ankle.confidence >= config.min_opposite_ankle_confidence and
                          in_frame(other_ankle))
        foot_reliable = (other_heel is not None and other_foot is not None and
                         other_heel.confidence >= config.min_opposite_foot_confidence and
                         other_foot.confidence >= config.min_opposite_foot_confidence and
                         in_frame(other_heel) and in_frame(other_foot))
        if not ankle_reliable and not foot_reliable:
            if any(p is not None and p.confidence >= config.min_opposite_foot_confidence and
                   not in_frame(p) for p in (other_ankle, other_heel, other_foot)):
                return self._result(side, reliability, "Move back so both feet are visible")
            return self._result(side, reliability, "Unable to reliably distinguish squat from lunge")
        stance_candidates = []
        guard_confidences = []
        if ankle_reliable:
            stance_candidates.append(abs(ankle.x-other_ankle.x)*width/torso_length)
            guard_confidences.append(other_ankle.confidence)
        if foot_reliable:
            stance_candidates.extend((abs(heel.x-other_heel.x)*width/torso_length,
                                      abs(foot.x-other_foot.x)*width/torso_length))
            guard_confidences.extend((other_heel.confidence, other_foot.confidence))
        stance_ratio = max(stance_candidates)
        stance_reliability = min(guard_confidences)
        stance_source = ("ankles+feet" if ankle_reliable and foot_reliable else
                         "ankles" if ankle_reliable else "feet")
        if stance_ratio > config.max_ankle_separation_to_torso:
            self._clear_attempt()
            return self._result(side, reliability,
                                "Feet are too far apart for a side-view squat",
                                stance_ratio=stance_ratio, stance_source=stance_source,
                                stance_reliability=stance_reliability)
        if all(name in landmarks and landmarks[name].confidence >= config.min_landmark_confidence
               for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            shoulder_width = abs(landmarks["left_shoulder"].x-landmarks["right_shoulder"].x)*width
            hip_width = abs(landmarks["left_hip"].x-landmarks["right_hip"].x)*width
            if max(shoulder_width, hip_width) / torso_length > config.max_side_width_to_torso:
                return self._result(side, reliability, "Turn sideways to the camera")

        try:
            raw = Angles(joint_angle(hip, knee, ankle, width, height),
                         joint_angle(shoulder, hip, knee, width, height),
                         lean_from_vertical(shoulder, hip, width, height))
        except ValueError:
            return self._result(side, reliability, "Unable to reliably assess")

        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > config.max_missing_seconds:
            self._clear_attempt()
        alpha = 1.0 if self.smoothed is None or gap is None or gap <= 0 else 1-math.exp(-gap/config.smoothing_tau_seconds)
        if self.smoothed is None:
            self.smoothed = raw
        else:
            self.smoothed = Angles(*(old + alpha*(new-old) for old, new in zip(
                (self.smoothed.knee, self.smoothed.hip, self.smoothed.torso),
                (raw.knee, raw.hip, raw.torso))))
        self.last_valid_time = timestamp
        angles = self.smoothed

        if self.state == State.STANDING:
            if angles.knee >= config.standing_knee_angle:
                self.ready = True
            if self.ready and angles.knee <= config.start_descending_knee_angle:
                self.state = State.DESCENDING
                self.attempt = {"start_time": timestamp, "minimum_knee_angle": angles.knee,
                                "minimum_hip_angle": angles.hip, "maximum_torso_lean": angles.torso,
                                "minimum_reliability": reliability,
                                "minimum_stance_reliability": stance_reliability}
        elif self.attempt is not None:
            a = self.attempt
            a["minimum_knee_angle"] = min(a["minimum_knee_angle"], angles.knee)
            a["minimum_hip_angle"] = min(a["minimum_hip_angle"], angles.hip)
            a["maximum_torso_lean"] = max(a["maximum_torso_lean"], angles.torso)
            a["minimum_reliability"] = min(a["minimum_reliability"], reliability)
            a["minimum_stance_reliability"] = min(a["minimum_stance_reliability"],
                                                  stance_reliability)
            if timestamp-a["start_time"] > config.max_rep_duration_seconds:
                self._clear_attempt()
            elif self.state == State.DESCENDING:
                if angles.knee <= config.bottom_entry_knee_angle:
                    self.state = State.BOTTOM
                elif angles.knee >= config.standing_knee_angle:
                    self._clear_attempt()
                    self.ready = True
            elif self.state == State.BOTTOM:
                if angles.knee >= a["minimum_knee_angle"] + config.ascent_from_bottom_degrees:
                    self.state = State.ASCENDING
            elif self.state == State.ASCENDING:
                if angles.knee >= config.standing_knee_angle:
                    self.reps.append({
                        "number": len(self.reps)+1,
                        "duration_seconds": round(timestamp-a["start_time"], 3),
                        "minimum_knee_angle": round(a["minimum_knee_angle"], 3),
                        "minimum_hip_angle": round(a["minimum_hip_angle"], 3),
                        "maximum_torso_lean": round(a["maximum_torso_lean"], 3),
                        "minimum_depth_reached": a["minimum_knee_angle"] <= config.depth_knee_angle,
                        "excessive_torso_lean": a["maximum_torso_lean"] >= config.excessive_torso_lean_degrees,
                        "minimum_landmark_reliability": round(a["minimum_reliability"], 3),
                        "minimum_stance_reliability": round(a["minimum_stance_reliability"], 3),
                    })
                    self.state = State.STANDING
                    self.attempt = None
                    self.ready = True
                elif angles.knee <= a["minimum_knee_angle"] + 2:
                    self.state = State.BOTTOM

        shallow_after_bottom = (self.state in (State.BOTTOM, State.ASCENDING) and
                                self.attempt is not None and
                                self.attempt["minimum_knee_angle"] > config.depth_knee_angle)
        lean_warning = angles.torso >= config.excessive_torso_lean_degrees
        form_warning = lean_warning or shallow_after_bottom
        if lean_warning:
            feedback = "Keep your torso more upright"
        elif self.state == State.ASCENDING:
            feedback = "Return to standing"
        elif self.state == State.BOTTOM and self.attempt and self.attempt["minimum_knee_angle"] <= config.depth_knee_angle:
            feedback = "Good depth"
        elif self.state in (State.DESCENDING, State.BOTTOM):
            feedback = "Go lower"
        else:
            feedback = "Ready for squat" if self.ready else "Stand upright to start"
        return self._result(side, reliability, feedback, True, form_warning,
                            stance_ratio, stance_source, stance_reliability)
