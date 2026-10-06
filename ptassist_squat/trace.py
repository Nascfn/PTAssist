"""Local, video-free frame diagnostics over generic pose landmarks."""

from __future__ import annotations

from dataclasses import asdict
import math

from .geometry import joint_angle, lean_from_vertical
from .landmarks import Landmarks


SQUAT_PARTS = ("nose", "shoulder", "hip", "knee", "ankle", "heel", "foot_index")


def frame_record(frame: int, timestamp: float, width: int, height: int,
                 landmarks: Landmarks, result: object, exercise: str,
                 config: object) -> dict:
    """Keep both measured output and the inputs needed to diagnose invalid frames."""
    selected_side = getattr(result, "side", None)
    raw_angles = None
    if exercise in ("squat", "toe-touch", "lunge") and selected_side:
        shoulder = landmarks.get(f"{selected_side}_shoulder")
        hip = landmarks.get(f"{selected_side}_hip")
        knee = landmarks.get(f"{selected_side}_knee")
        ankle = landmarks.get(f"{selected_side}_ankle")
        if all((shoulder, hip, knee, ankle)):
            try:
                raw_angles = {
                    "knee": round(joint_angle(hip, knee, ankle, width, height), 3),
                    "torso": round(lean_from_vertical(shoulder, hip, width, height), 3),
                }
                if exercise == "squat":
                    raw_angles["hip"] = round(joint_angle(shoulder, hip, knee, width, height), 3)
                elif exercise == "toe-touch":
                    finger = landmarks.get(f"{selected_side}_index")
                    toe = landmarks.get(f"{selected_side}_foot_index")
                    leg_length = (math.hypot((hip.x-knee.x)*width, (hip.y-knee.y)*height) +
                                  math.hypot((knee.x-ankle.x)*width, (knee.y-ankle.y)*height))
                    if finger and toe and leg_length >= 30:
                        raw_angles["finger_to_toe_leg_lengths"] = round(math.hypot(
                            (finger.x-toe.x)*width, (finger.y-toe.y)*height)/leg_length, 3)
            except ValueError:
                pass
    if exercise == "lunge" and raw_angles is not None:
        other = "left" if selected_side == "right" else "right"
        other_hip = landmarks.get(f"{other}_hip")
        other_knee = landmarks.get(f"{other}_knee")
        other_ankle = landmarks.get(f"{other}_ankle")
        if all((other_hip, other_knee, other_ankle)):
            try:
                raw_angles[f"{other}_knee"] = round(joint_angle(
                    other_hip, other_knee, other_ankle, width, height), 3)
                raw_angles[f"{selected_side}_knee"] = raw_angles.pop("knee")
            except ValueError:
                pass

    issues = []
    if exercise == "squat":
        required = ["nose"]
        if selected_side:
            required += [f"{selected_side}_{part}" for part in SQUAT_PARTS if part != "nose"]
            other = "left" if selected_side == "right" else "right"
            required += [f"{other}_{part}" for part in ("ankle", "heel", "foot_index")]
    elif exercise == "toe-touch":
        required = ([f"{selected_side}_{part}" for part in
                     ("shoulder", "hip", "knee", "ankle", "index", "foot_index")]
                    if selected_side else [])
    elif exercise == "lunge":
        required = ([f"{leg}_{part}" for leg in ("left", "right")
                     for part in ("hip", "knee", "ankle")]
                    + ([f"{selected_side}_shoulder"] if selected_side else []))
    elif exercise == "shoulder-abduction":
        required = [f"{side}_{part}" for side in ("left", "right")
                    for part in ("shoulder", "hip")]
        if selected_side:
            required += [f"{selected_side}_{part}" for part in ("elbow", "wrist")]
    elif exercise == "elbow-bend":
        required = ([f"{selected_side}_{part}" for part in
                     ("shoulder", "elbow", "wrist")] if selected_side else [])
    else:
        required = ["left_shoulder", "right_shoulder"]
        if selected_side:
            required.append(f"{selected_side}_wrist")
    for name in required:
        point = landmarks.get(name)
        threshold = config.min_landmark_confidence
        if exercise == "squat" and selected_side:
            if name == f"{other}_ankle":
                threshold = config.min_opposite_ankle_confidence
            elif name in (f"{other}_heel", f"{other}_foot_index"):
                threshold = config.min_opposite_foot_confidence
        if point is None:
            issues.append(f"{name}:missing")
        elif point.confidence < threshold:
            issues.append(f"{name}:low_confidence({point.confidence:.2f})")
        elif not (config.frame_margin <= point.x <= 1-config.frame_margin and
                  config.frame_margin <= point.y <= 1-config.frame_margin):
            issues.append(f"{name}:outside_frame")

    return {
        "type": "frame", "frame": frame, "timestamp": timestamp,
        "width": width, "height": height,
        "analysis": asdict(result), "raw_angles_diagnostic_only": raw_angles,
        "landmark_issues": issues,
        "landmarks": {name: [point.x, point.y, point.visibility, point.presence]
                      for name, point in landmarks.items()},
    }
