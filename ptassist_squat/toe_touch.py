"""Standing side-view toe-touch analysis over generic normalized landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .geometry import joint_angle, lean_from_vertical
from .landmarks import Landmarks


REQUIRED = ("shoulder", "hip", "knee", "ankle", "index", "foot_index")


class ToeTouchState(str, Enum):
    STANDING = "STANDING"
    BENDING = "BENDING"
    BOTTOM = "BOTTOM"
    RETURNING = "RETURNING"


@dataclass(frozen=True)
class ToeTouchConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_landmark_confidence: float = 0.65
    frame_margin: float = 0.025
    max_side_width_to_torso: float = 0.75
    side_switch_advantage: float = 0.12
    smoothing_tau_seconds: float = 0.12
    max_missing_seconds: float = 0.75
    upright_torso_angle: float = 15.0
    start_bending_torso_angle: float = 25.0
    bottom_torso_angle: float = 50.0
    return_from_bottom_degrees: float = 10.0
    minimum_start_knee_angle: float = 155.0
    noticeable_knee_flexion_degrees: float = 35.0
    near_toes_distance_ratio: float = 0.15
    max_rep_duration_seconds: float = 20.0


@dataclass
class ToeTouchMeasurements:
    knee: float
    torso: float
    toe_distance: float
    knee_flexion: float


@dataclass
class ToeTouchFrameResult:
    state: str
    rep_count: int
    side: str | None
    reliability: float
    measurements: ToeTouchMeasurements | None
    feedback: str
    assessment_valid: bool
    form_warning: bool = False


class ToeTouchAnalyzer:
    def __init__(self, config: ToeTouchConfig) -> None:
        self.config = config
        self.state = ToeTouchState.STANDING
        self.side: str | None = None
        self.smoothed: ToeTouchMeasurements | None = None
        self.last_valid_time: float | None = None
        self.standing_knee: float | None = None
        self.ready = False
        self.attempt: dict | None = None
        self.reps: list[dict] = []

    def _reset(self) -> None:
        self.state = ToeTouchState.STANDING
        self.smoothed = None
        self.standing_knee = None
        self.ready = False
        self.attempt = None

    def _result(self, reliability: float, feedback: str, valid: bool = False,
                warning: bool = False) -> ToeTouchFrameResult:
        return ToeTouchFrameResult(self.state.value, len(self.reps), self.side,
                                   reliability, self.smoothed if valid else None,
                                   feedback, valid, warning)

    def _missing(self, timestamp: float, reliability: float,
                 feedback: str = "Unable to reliably assess") -> ToeTouchFrameResult:
        if (self.last_valid_time is not None and
                timestamp-self.last_valid_time > self.config.max_missing_seconds):
            self._reset()
        return self._result(reliability, feedback)

    def _choose_side(self, landmarks: Landmarks) -> tuple[str | None, float]:
        scores = {}
        for side in ("left", "right"):
            points = [landmarks.get(f"{side}_{part}") for part in REQUIRED]
            scores[side] = (0.7*min(p.confidence for p in points) +
                            0.3*sum(p.confidence for p in points)/len(points)) if all(points) else 0.0
        candidate = max(scores, key=scores.get)
        if self.side and scores[self.side]+self.config.side_switch_advantage >= scores[candidate]:
            candidate = self.side
        points = [landmarks.get(f"{candidate}_{part}") for part in REQUIRED]
        reliability = min(p.confidence for p in points) if all(points) else 0.0
        return candidate, reliability

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> ToeTouchFrameResult:
        c = self.config
        side, reliability = self._choose_side(landmarks) if landmarks else (None, 0.0)
        if side is None or reliability < c.min_landmark_confidence:
            self.side = side
            return self._missing(timestamp, reliability)
        if side != self.side:
            self._reset()
        self.side = side
        selected = [landmarks[f"{side}_{part}"] for part in REQUIRED]
        if any(not (c.frame_margin <= p.x <= 1-c.frame_margin and
                    c.frame_margin <= p.y <= 1-c.frame_margin) for p in selected):
            return self._missing(timestamp, reliability,
                                 "Move back so your hand and feet are visible")
        nose = landmarks.get("nose")
        if nose and nose.confidence >= c.min_landmark_confidence and not (
                c.frame_margin <= nose.x <= 1-c.frame_margin and
                c.frame_margin <= nose.y <= 1-c.frame_margin):
            return self._missing(timestamp, reliability,
                                 "Move back so your full body is visible")
        shoulder, hip, knee, ankle, finger, toe = selected
        torso_length = math.hypot((shoulder.x-hip.x)*width, (shoulder.y-hip.y)*height)
        if torso_length < 15:
            return self._missing(timestamp, reliability)
        if all(name in landmarks and landmarks[name].confidence >= c.min_landmark_confidence
               for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            shoulder_width = abs(landmarks["left_shoulder"].x-landmarks["right_shoulder"].x)*width
            hip_width = abs(landmarks["left_hip"].x-landmarks["right_hip"].x)*width
            if max(shoulder_width, hip_width)/torso_length > c.max_side_width_to_torso:
                return self._missing(timestamp, reliability, "Turn sideways to the camera")
        try:
            knee_angle = joint_angle(hip, knee, ankle, width, height)
            torso_angle = lean_from_vertical(shoulder, hip, width, height)
        except ValueError:
            return self._missing(timestamp, reliability)
        leg_length = (math.hypot((hip.x-knee.x)*width, (hip.y-knee.y)*height) +
                      math.hypot((knee.x-ankle.x)*width, (knee.y-ankle.y)*height))
        if leg_length < 30:
            return self._missing(timestamp, reliability)
        toe_distance = math.hypot((finger.x-toe.x)*width,
                                  (finger.y-toe.y)*height)/leg_length

        gap = None if self.last_valid_time is None else timestamp-self.last_valid_time
        if gap is not None and gap > c.max_missing_seconds:
            self._reset()
            self.side = side
        alpha = (1.0 if self.smoothed is None or gap is None or gap <= 0 else
                 1-math.exp(-gap/c.smoothing_tau_seconds))
        if self.smoothed:
            knee_angle = self.smoothed.knee+alpha*(knee_angle-self.smoothed.knee)
            torso_angle = self.smoothed.torso+alpha*(torso_angle-self.smoothed.torso)
            toe_distance = self.smoothed.toe_distance+alpha*(toe_distance-self.smoothed.toe_distance)
        self.last_valid_time = timestamp
        baseline = self.standing_knee if self.standing_knee is not None else knee_angle
        knee_flexion = max(0.0, baseline-knee_angle)
        self.smoothed = ToeTouchMeasurements(knee_angle, torso_angle, toe_distance,
                                              knee_flexion)

        if self.state == ToeTouchState.STANDING:
            if torso_angle <= c.upright_torso_angle and knee_angle >= c.minimum_start_knee_angle:
                self.ready = True
                self.standing_knee = max(self.standing_knee or knee_angle, knee_angle)
            if self.ready and torso_angle >= c.start_bending_torso_angle:
                self.state = ToeTouchState.BENDING
                self.attempt = {"start_time": timestamp,
                                "closest_toe_distance": toe_distance,
                                "minimum_knee_angle": knee_angle,
                                "maximum_knee_flexion": knee_flexion,
                                "maximum_torso_lean": torso_angle,
                                "minimum_reliability": reliability}
        elif self.attempt:
            a = self.attempt
            a["closest_toe_distance"] = min(a["closest_toe_distance"], toe_distance)
            a["minimum_knee_angle"] = min(a["minimum_knee_angle"], knee_angle)
            a["maximum_knee_flexion"] = max(a["maximum_knee_flexion"], knee_flexion)
            a["maximum_torso_lean"] = max(a["maximum_torso_lean"], torso_angle)
            a["minimum_reliability"] = min(a["minimum_reliability"], reliability)
            if timestamp-a["start_time"] > c.max_rep_duration_seconds:
                self.state = ToeTouchState.STANDING
                self.attempt = None
                self.standing_knee = None
                self.ready = False
            elif self.state == ToeTouchState.BENDING:
                if torso_angle >= c.bottom_torso_angle:
                    self.state = ToeTouchState.BOTTOM
                elif torso_angle <= c.upright_torso_angle:
                    self.state = ToeTouchState.STANDING
                    self.attempt = None
                    self.standing_knee = knee_angle
                    self.ready = True
            elif self.state == ToeTouchState.BOTTOM:
                if torso_angle <= a["maximum_torso_lean"]-c.return_from_bottom_degrees:
                    self.state = ToeTouchState.RETURNING
            elif self.state == ToeTouchState.RETURNING:
                if torso_angle <= c.upright_torso_angle:
                    self.reps.append({
                        "number": len(self.reps)+1,
                        "duration_seconds": round(timestamp-a["start_time"], 3),
                        "closest_toe_distance_leg_lengths": round(a["closest_toe_distance"], 3),
                        "minimum_knee_angle": round(a["minimum_knee_angle"], 3),
                        "maximum_knee_flexion": round(a["maximum_knee_flexion"], 3),
                        "maximum_torso_lean": round(a["maximum_torso_lean"], 3),
                        "near_toes_2d": a["closest_toe_distance"] <= c.near_toes_distance_ratio,
                        "noticeable_knee_bend": a["maximum_knee_flexion"] >= c.noticeable_knee_flexion_degrees,
                        "minimum_landmark_reliability": round(a["minimum_reliability"], 3),
                    })
                    self.state = ToeTouchState.STANDING
                    self.attempt = None
                    self.standing_knee = knee_angle
                    self.ready = True
                elif torso_angle >= a["maximum_torso_lean"]-2:
                    self.state = ToeTouchState.BOTTOM

        warning = (self.attempt is not None and
                   knee_flexion >= c.noticeable_knee_flexion_degrees)
        if warning:
            feedback = "Noticeable knee bend"
        elif self.state == ToeTouchState.BOTTOM and toe_distance <= c.near_toes_distance_ratio:
            feedback = "Near your toes in the camera view"
        elif self.state == ToeTouchState.RETURNING:
            feedback = "Return to standing"
        elif self.state in (ToeTouchState.BENDING, ToeTouchState.BOTTOM):
            feedback = "Reach toward your toes"
        else:
            feedback = "Ready for toe touch" if self.ready else "Stand upright to start"
        return self._result(reliability, feedback, True, warning)
