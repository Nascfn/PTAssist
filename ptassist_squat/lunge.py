"""Side-view stepping lunge state machine over generic pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .geometry import joint_angle, lean_from_vertical
from .landmarks import Landmarks


class LungeState(str, Enum):
    STANDING = "STANDING"
    STEPPING = "STEPPING"
    BOTTOM = "BOTTOM"
    RISING = "RISING"


@dataclass(frozen=True)
class LungeConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_landmark_confidence: float = 0.6
    frame_margin: float = 0.025
    max_side_width_to_torso: float = 0.75
    side_switch_advantage: float = 0.12
    smoothing_tau_seconds: float = 0.12
    max_missing_seconds: float = 0.75
    standing_stance_ratio: float = 0.6
    step_start_stance_ratio: float = 0.8
    min_step_displacement_ratio: float = 0.35
    min_step_asymmetry_ratio: float = 0.15
    standing_knee_angle: float = 160.0
    bottom_entry_knee_angle: float = 140.0
    ascent_from_bottom_degrees: float = 8.0
    depth_knee_angle: float = 110.0
    excessive_torso_lean_degrees: float = 45.0
    max_rep_duration_seconds: float = 15.0
    reference_knee_tolerance_degrees: float = 12.0
    reference_torso_tolerance_degrees: float = 10.0
    reference_duration_tolerance_fraction: float = 0.4
    reference_stance_tolerance_fraction: float = 0.2


@dataclass
class LungeMeasurements:
    left_knee: float
    right_knee: float
    torso: float
    stance_ratio: float


@dataclass
class LungeFrameResult:
    state: str
    rep_count: int
    side: str | None
    lead_side: str | None
    reliability: float
    measurements: LungeMeasurements | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False


class LungeAnalyzer:
    def __init__(self, config: LungeConfig) -> None:
        self.config = config
        self.state = LungeState.STANDING
        self.side: str | None = None
        self.lead_side: str | None = None
        self.smoothed: LungeMeasurements | None = None
        self.last_valid_time: float | None = None
        self.baseline_ankles: dict[str, float] | None = None
        self.ready = False
        self.attempt: dict | None = None
        self.reps: list[dict] = []

    def _cancel(self) -> None:
        self.state = LungeState.STANDING
        self.lead_side = None
        self.baseline_ankles = None
        self.ready = False
        self.attempt = None

    def _reset(self) -> None:
        self._cancel()
        self.smoothed = None

    def _result(self, reliability: float, feedback: str, valid: bool = False,
                warning: bool = False) -> LungeFrameResult:
        return LungeFrameResult(self.state.value, len(self.reps), self.side,
                                self.lead_side, reliability,
                                self.smoothed if valid else None,
                                feedback, valid, warning)

    def _missing(self, timestamp: float, reliability: float,
                 feedback: str = "Unable to reliably assess") -> LungeFrameResult:
        if (self.last_valid_time is not None and
                timestamp-self.last_valid_time > self.config.max_missing_seconds):
            self._reset()
        return self._result(reliability, feedback)

    def _choose_torso_side(self, landmarks: Landmarks) -> tuple[str | None, float]:
        scores = {}
        for side in ("left", "right"):
            shoulder = landmarks.get(f"{side}_shoulder")
            hip = landmarks.get(f"{side}_hip")
            scores[side] = min(shoulder.confidence, hip.confidence) if shoulder and hip else 0.0
        candidate = max(scores, key=scores.get)
        if self.side and scores[self.side]+self.config.side_switch_advantage >= scores[candidate]:
            candidate = self.side
        return candidate, scores[candidate]

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> LungeFrameResult:
        c = self.config
        side, torso_confidence = self._choose_torso_side(landmarks) if landmarks else (None, 0.0)
        self.side = side
        required = [f"{leg}_{part}" for leg in ("left", "right")
                    for part in ("hip", "knee", "ankle")]
        if side:
            required.append(f"{side}_shoulder")
        points = [landmarks.get(name) for name in required]
        reliability = min((p.confidence for p in points), default=0.0) if all(points) else 0.0
        reliability = min(reliability, torso_confidence)
        if side is None or reliability < c.min_landmark_confidence:
            return self._missing(timestamp, reliability)
        if any(not (c.frame_margin <= p.x <= 1-c.frame_margin and
                    c.frame_margin <= p.y <= 1-c.frame_margin) for p in points):
            return self._missing(timestamp, reliability,
                                 "Move back so both legs and feet are visible")
        shoulder = landmarks[f"{side}_shoulder"]
        hip = landmarks[f"{side}_hip"]
        torso_length = math.hypot((shoulder.x-hip.x)*width,
                                  (shoulder.y-hip.y)*height)
        if torso_length < 15:
            return self._missing(timestamp, reliability)
        if all(name in landmarks and landmarks[name].confidence >= c.min_landmark_confidence
               for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            shoulder_width = abs(landmarks["left_shoulder"].x-landmarks["right_shoulder"].x)*width
            hip_width = abs(landmarks["left_hip"].x-landmarks["right_hip"].x)*width
            if max(shoulder_width, hip_width)/torso_length > c.max_side_width_to_torso:
                return self._missing(timestamp, reliability, "Turn sideways to the camera")
        try:
            left_knee = joint_angle(landmarks["left_hip"], landmarks["left_knee"],
                                    landmarks["left_ankle"], width, height)
            right_knee = joint_angle(landmarks["right_hip"], landmarks["right_knee"],
                                     landmarks["right_ankle"], width, height)
            torso = lean_from_vertical(shoulder, hip, width, height)
        except ValueError:
            return self._missing(timestamp, reliability)
        stance = abs(landmarks["left_ankle"].x-landmarks["right_ankle"].x)*width/torso_length
        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > c.max_missing_seconds:
            self._reset()
        alpha = (1.0 if self.smoothed is None or gap is None or gap <= 0 else
                 1-math.exp(-gap/c.smoothing_tau_seconds))
        if self.smoothed:
            left_knee = self.smoothed.left_knee+alpha*(left_knee-self.smoothed.left_knee)
            right_knee = self.smoothed.right_knee+alpha*(right_knee-self.smoothed.right_knee)
            torso = self.smoothed.torso+alpha*(torso-self.smoothed.torso)
            stance = self.smoothed.stance_ratio+alpha*(stance-self.smoothed.stance_ratio)
        self.smoothed = LungeMeasurements(left_knee, right_knee, torso, stance)
        self.last_valid_time = timestamp
        knee_angles = {"left": left_knee, "right": right_knee}

        if self.state == LungeState.STANDING:
            if (stance <= c.standing_stance_ratio and
                    min(knee_angles.values()) >= c.standing_knee_angle):
                self.ready = True
                self.baseline_ankles = {leg: landmarks[f"{leg}_ankle"].x
                                        for leg in ("left", "right")}
            if self.ready and self.baseline_ankles and stance >= c.step_start_stance_ratio:
                movements = {leg: abs(landmarks[f"{leg}_ankle"].x-
                                      self.baseline_ankles[leg])*width/torso_length
                             for leg in ("left", "right")}
                lead = max(movements, key=movements.get)
                other = "right" if lead == "left" else "left"
                if (movements[lead] >= c.min_step_displacement_ratio and
                        movements[lead]-movements[other] >= c.min_step_asymmetry_ratio):
                    self.lead_side = lead
                    self.state = LungeState.STEPPING
                    self.attempt = {
                        "start_time": timestamp,
                        "minimum_lead_knee_angle": knee_angles[lead],
                        "minimum_trail_knee_angle": knee_angles[other],
                        "maximum_torso_lean": torso,
                        "maximum_stance_ratio": stance,
                        "minimum_reliability": reliability,
                    }
        elif self.attempt and self.lead_side:
            a = self.attempt
            lead = self.lead_side
            other = "right" if lead == "left" else "left"
            a["minimum_lead_knee_angle"] = min(a["minimum_lead_knee_angle"], knee_angles[lead])
            a["minimum_trail_knee_angle"] = min(a["minimum_trail_knee_angle"], knee_angles[other])
            a["maximum_torso_lean"] = max(a["maximum_torso_lean"], torso)
            a["maximum_stance_ratio"] = max(a["maximum_stance_ratio"], stance)
            a["minimum_reliability"] = min(a["minimum_reliability"], reliability)
            if timestamp-a["start_time"] > c.max_rep_duration_seconds:
                self._cancel()
            elif self.state == LungeState.STEPPING:
                if stance <= c.standing_stance_ratio:
                    self._cancel()
                elif knee_angles[lead] <= c.bottom_entry_knee_angle:
                    self.state = LungeState.BOTTOM
            elif self.state == LungeState.BOTTOM:
                if knee_angles[lead] >= a["minimum_lead_knee_angle"]+c.ascent_from_bottom_degrees:
                    self.state = LungeState.RISING
            elif self.state == LungeState.RISING:
                if (stance <= c.standing_stance_ratio and
                        min(knee_angles.values()) >= c.standing_knee_angle):
                    self.reps.append({
                        "number": len(self.reps)+1,
                        "lead_side": lead,
                        "duration_seconds": round(timestamp-a["start_time"], 3),
                        "minimum_lead_knee_angle": round(a["minimum_lead_knee_angle"], 3),
                        "minimum_trail_knee_angle": round(a["minimum_trail_knee_angle"], 3),
                        "maximum_torso_lean": round(a["maximum_torso_lean"], 3),
                        "maximum_stance_ratio": round(a["maximum_stance_ratio"], 3),
                        "minimum_depth_reached": a["minimum_lead_knee_angle"] <= c.depth_knee_angle,
                        "excessive_torso_lean": a["maximum_torso_lean"] >= c.excessive_torso_lean_degrees,
                        "minimum_landmark_reliability": round(a["minimum_reliability"], 3),
                    })
                    self._cancel()
                    self.ready = True
                    self.baseline_ankles = {leg: landmarks[f"{leg}_ankle"].x
                                            for leg in ("left", "right")}
                elif knee_angles[lead] <= a["minimum_lead_knee_angle"]+2:
                    self.state = LungeState.BOTTOM

        lean_warning = torso >= c.excessive_torso_lean_degrees
        if lean_warning:
            feedback = "Torso lean above demo threshold"
        elif self.state == LungeState.RISING:
            feedback = "Rise and step back"
        elif self.state == LungeState.BOTTOM:
            feedback = "Rise from the lunge"
        elif self.state == LungeState.STEPPING:
            feedback = "Bend the stepping knee"
        else:
            feedback = "Step forward for a lunge" if self.ready else "Stand with feet together to start"
        return self._result(reliability, feedback, True, lean_warning)
