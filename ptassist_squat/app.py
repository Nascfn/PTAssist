"""Local camera/video UI and session writer."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .analysis import Config, SquatAnalyzer
from .arm_stretch import ArmStretchConfig, CrossBodyStretchAnalyzer
from .elbow_bend import ElbowBendAnalyzer, ElbowConfig
from .lunge import LungeAnalyzer, LungeConfig
from .pose import PoseEstimator, draw_skeleton
from .reference import ComparisonTolerances, ReferenceCoach
from .review import ReviewClip, play_review
from .shoulder_abduction import AbductionConfig, ShoulderAbductionAnalyzer
from .toe_touch import ToeTouchAnalyzer, ToeTouchConfig
from .trace import frame_record


ROOT = Path(__file__).resolve().parent.parent
WINDOW_NAME = "PTAssist exercise analysis"
GREEN = (60, 220, 70)
ORANGE = (0, 165, 255)
RED = (60, 60, 255)


def screen_size() -> tuple[int, int]:
    """Use the physical desktop size so a fullscreen image fills high-DPI displays."""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def window_size(fallback: tuple[int, int]) -> tuple[int, int]:
    """Read the drawable area after a user resizes the OpenCV window."""
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return fallback
    return (width, height) if width > 0 and height > 0 else fallback


def fit_to_window(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale and center the camera image without stretching body proportions."""
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    fitted_width = max(1, round(source_width * scale))
    fitted_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
    fitted = cv2.resize(frame, (fitted_width, fitted_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - fitted_width) // 2
    y = (height - fitted_height) // 2
    canvas[y:y+fitted_height, x:x+fitted_width] = fitted
    return canvas


def display_color(result, flash_on: bool) -> tuple[int, int, int] | None:
    if not result.assessment_valid:
        return RED
    if result.form_warning:
        return ORANGE if flash_on else None
    return GREEN


def wrap_overlay_line(line: str, max_width: int, scale: float = 0.5) -> list[str]:
    """Keep status and reference text inside the camera image before scaling it."""
    words = line.split()
    if not words:
        return [""]
    wrapped = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= max_width:
            current = candidate
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def handle_reference_key(key: int, coach: ReferenceCoach | None) -> Path | None:
    if key != ord("a") or coach is None:
        return None
    if coach.last_rep is None:
        coach.last_message = "Complete a rep before pressing A"
        return None
    return coach.approve_last()


def draw_overlay(frame, result, fps: float, flash_on: bool, exercise: str = "squat",
                 coach: ReferenceCoach | None = None,
                 review_enabled: bool = True) -> None:
    if exercise == "squat":
        count_label = "Reps"
        foot_source = f" ({result.stance_source})" if result.stance_source else ""
        measurements = (f"Knee: {result.angles.knee:.1f}  Hip: {result.angles.hip:.1f}  "
                        f"Torso: {result.angles.torso:.1f}  "
                        f"Feet: {result.stance_ratio:.2f}{foot_source}") if result.angles else (
                            f"Knee: --  Hip: --  Torso: --  Feet: {result.stance_ratio:.2f}{foot_source}"
                            if result.stance_ratio is not None else "Knee: --  Hip: --  Torso: --  Feet: --")
        view_hint = "Side view | full body"
        extra_line = None
    elif exercise == "cross-body-stretch":
        count_label = "Cycles"
        measurements = (f"Cross-body reach: {result.metrics.reach*100:.0f}%  "
                        f"Hold: {result.metrics.hold_seconds:.1f}s") if result.metrics else "Cross-body reach: --  Hold: --"
        view_hint = "Front view | stretching wrist visible"
        extra_line = "Arm straightness: unassessed when arms overlap"
    elif exercise == "shoulder-abduction":
        count_label = "Cycles"
        measurements = (f"Shoulder: {result.metrics.shoulder_angle:.0f} deg  "
                        f"Hand arc: {result.metrics.hand_angle:.0f} deg  "
                        f"Elbow: {result.metrics.elbow_angle:.0f} deg  "
                        f"Torso: {result.metrics.torso_lean:.0f} deg  "
                        f"Full arcs: {result.full_arc_count}") if result.metrics else (
                            f"Shoulder: --  Hand arc: --  Elbow: --  Torso: --  "
                            f"Full arcs: {result.full_arc_count}")
        view_hint = "Front view | one arm at a time | side-to-overhead"
        extra_line = "180 deg is a 2D target; >=165 deg counts as near-overhead"
    elif exercise == "elbow-bend":
        count_label = "Cycles"
        measurements = (f"Elbow: {result.metrics.elbow_angle:.0f} deg  "
                        f"Upper arm: {result.metrics.upper_arm_angle:.0f} deg  "
                        f"Full bends: {result.full_bend_count}") if result.metrics else (
                            f"Elbow: --  Upper arm: --  Full bends: {result.full_bend_count}")
        view_hint = "Upper body visible | one arm at a time"
        extra_line = "Bend and straighten; 95 deg is an experimental POC target"
    elif exercise == "toe-touch":
        count_label = "Reps"
        measurements = (f"Knee: {result.measurements.knee:.1f}  "
                        f"Torso: {result.measurements.torso:.1f}  "
                        f"Finger-to-toe: {result.measurements.toe_distance:.2f} leg lengths  "
                        f"Knee bend: {result.measurements.knee_flexion:.1f}") if result.measurements else (
                            "Knee: --  Torso: --  Finger-to-toe: --  Knee bend: --")
        view_hint = "Side view | hand and feet visible"
        extra_line = "Finger-to-toe is a 2D estimate, not confirmed contact"
    else:
        count_label = "Reps"
        measurements = (f"Left knee: {result.measurements.left_knee:.1f}  "
                        f"Right knee: {result.measurements.right_knee:.1f}  "
                        f"Torso: {result.measurements.torso:.1f}  "
                        f"Stance: {result.measurements.stance_ratio:.2f}") if result.measurements else (
                            "Left knee: --  Right knee: --  Torso: --  Stance: --")
        view_hint = "Side view | step out, bend, rise, step back"
        extra_line = f"Lead leg: {result.lead_side or '-'}"
    confidence_line = (f"Selected side: {result.side or '-'} | "
                       f"Pose confidence: {result.reliability:.2f}")
    if exercise == "squat":
        foot_confidence = (f"{result.stance_reliability:.2f}"
                           if result.stance_reliability is not None else "--")
        confidence_line += f" | Foot guard confidence: {foot_confidence}"
    lines = [
        f"{exercise}: {result.state} | {count_label}: {result.rep_count} | FPS: {fps:.1f}",
        confidence_line,
        measurements,
        f"Feedback: {result.feedback}",
    ]
    if extra_line:
        lines.append(extra_line)
    reference_line_index = None
    reference_color = None
    if coach:
        if coach.last_rep:
            rep = coach.last_rep
            if exercise == "squat":
                lines.append(f"Last rep: knee {rep['minimum_knee_angle']:.1f}  "
                             f"hip {rep['minimum_hip_angle']:.1f}  "
                             f"torso {rep['maximum_torso_lean']:.1f}  "
                             f"time {rep['duration_seconds']:.1f}s")
            else:
                lines.append(f"Last rep: {rep['lead_side']} lead knee {rep['minimum_lead_knee_angle']:.1f}  "
                             f"torso {rep['maximum_torso_lean']:.1f}  "
                             f"stance {rep['maximum_stance_ratio']:.2f}  "
                             f"time {rep['duration_seconds']:.1f}s")
        reference_line_index = len(lines)
        reference_message = coach.message_for_frame(result)
        lines.append(f"Reference: {reference_message}")
        if not result.assessment_valid:
            reference_color = RED
        elif reference_message.startswith(("Torso leaning", "Bend deeper", "Bend lead", "Step appears")):
            reference_color = ORANGE
        elif result.state == "STANDING" and coach.last_rep:
            comparison = coach.last_rep.get("reference_comparison")
            if comparison and not comparison["within_reference_tolerances"]:
                reference_color = ORANGE
    replay_hint = " | V: replay last rep" if review_enabled else ""
    lines.append(f"{view_hint} | F: fullscreen{replay_hint} | Q or Esc: save and quit")
    if coach:
        lines.append("A: approve latest rep as reference (replaces that side's reference)")
    width = frame.shape[1]
    top_count = (5 if extra_line else 4) if coach else len(lines)
    top_lines = []
    bottom_lines = []
    for index, line in enumerate(lines):
        for segment in wrap_overlay_line(line, max(100, width-20)):
            (top_lines if index < top_count else bottom_lines).append((index, segment))
    overlay = frame.copy()
    top_height = 25*len(top_lines)+2
    bottom_start = frame.shape[0]-25*len(bottom_lines)-2
    cv2.rectangle(overlay, (0, 0), (width, top_height), (0, 0, 0), -1)
    if bottom_lines:
        cv2.rectangle(overlay, (0, bottom_start), (width, frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for section, start in ((top_lines, 0), (bottom_lines, bottom_start)):
        for row, (index, line) in enumerate(section):
            color = (255, 255, 255)
            if index == 3:
                color = display_color(result, flash_on) or (255, 255, 255)
            elif index == reference_line_index and reference_color:
                color = reference_color
            cv2.putText(frame, line, (10, start + 21 + 25*row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)


def draw_status_border(frame: np.ndarray, result, flash_on: bool) -> None:
    color = display_color(result, flash_on)
    if color is not None:
        height, width = frame.shape[:2]
        thickness = max(6, round(min(width, height) * 0.009))
        cv2.rectangle(frame, (0, 0), (width-1, height-1), color, thickness)


def draw_issue_markers(frame: np.ndarray, landmarks, result, exercise: str,
                       config) -> dict[str, float]:
    """Mark the measured body part behind a live warning, only on reliable frames."""
    if not result.assessment_valid:
        return {}
    side = getattr(result, "side", None)
    if side not in ("left", "right"):
        return {}
    height, width = frame.shape[:2]

    def pixel(name: str):
        p = landmarks.get(name)
        if p is None or p.confidence < config.min_landmark_confidence:
            return None
        if not (0 <= p.x <= 1 and 0 <= p.y <= 1):
            return None
        return round(p.x*width), round(p.y*height)

    feedback = result.feedback.lower()
    issues = {}
    if exercise == "shoulder-abduction":
        shoulder = pixel(f"{side}_shoulder")
        elbow = pixel(f"{side}_elbow")
        wrist = pixel(f"{side}_wrist")
        left_hip, right_hip = pixel("left_hip"), pixel("right_hip")
        left_shoulder, right_shoulder = pixel("left_shoulder"), pixel("right_shoulder")
        if not all((shoulder, elbow, wrist, left_hip, right_hip,
                    left_shoulder, right_shoulder)):
            return issues
        hip_mid = np.mean((left_hip, right_hip), axis=0)
        shoulder_mid = np.mean((left_shoulder, right_shoulder), axis=0)
        torso = hip_mid-shoulder_mid
        # Draw a short angular cue only while a raise or return is in progress.
        # The shoulder angle decides its direction and when it disappears;
        # wrist distance to a fixed screen point never controls this cue.
        if (result.metrics and
                ((result.state == "RAISING" and
                  result.metrics.shoulder_angle < config.target_angle_degrees) or
                 (result.state == "LOWERING" and
                  result.metrics.shoulder_angle > config.return_angle_degrees))):
            angle = result.metrics.shoulder_angle
            if "torso upright" in feedback:
                cv2.line(frame, tuple(np.asarray(hip_mid, dtype=int)),
                         tuple(np.asarray(shoulder_mid, dtype=int)), ORANGE, 3,
                         cv2.LINE_AA)
            elif "straighten your elbow" in feedback:
                upper = np.asarray(elbow, dtype=float)-shoulder
                forearm = np.linalg.norm(np.asarray(wrist)-elbow)
                target = np.asarray(elbow)+upper/max(np.linalg.norm(upper), 1)*forearm
                vector = target-wrist
                if np.linalg.norm(vector) > 10:
                    arrow_end = np.asarray(wrist)+vector/max(np.linalg.norm(vector), 1)*min(
                        np.linalg.norm(vector), 75)
                    cv2.arrowedLine(frame, wrist, tuple(np.rint(arrow_end).astype(int)),
                                    ORANGE, 5, cv2.LINE_AA, tipLength=.24)
                    cv2.circle(frame, elbow, 11, ORANGE, 2, cv2.LINE_AA)
            elif ((result.state == "RAISING" and angle < config.target_angle_degrees)
                  or (result.state == "LOWERING" and angle > config.return_angle_degrees)):
                down = torso/max(np.linalg.norm(torso), 1)
                outward = np.array((down[1], -down[0]))
                other_shoulder = right_shoulder if side == "left" else left_shoulder
                if shoulder[0] < other_shoulder[0]:
                    outward *= -1
                desired_angle = np.clip(angle+(25 if result.state == "RAISING" else -25),
                                        0, 180)
                radians = np.deg2rad(desired_angle)
                direction = down*np.cos(radians)+outward*np.sin(radians)
                upper_length = np.linalg.norm(np.asarray(elbow)-shoulder)
                target = np.asarray(shoulder)+direction*upper_length
                vector = target-elbow
                if np.linalg.norm(vector) > 10:
                    arrow_end = np.asarray(elbow)+vector/max(np.linalg.norm(vector), 1)*min(
                        np.linalg.norm(vector), 75)
                    cv2.arrowedLine(frame, elbow, tuple(np.rint(arrow_end).astype(int)),
                                    (20, 20, 20), 10, cv2.LINE_AA, tipLength=.25)
                    cv2.arrowedLine(frame, elbow, tuple(np.rint(arrow_end).astype(int)),
                                    ORANGE, 5, cv2.LINE_AA, tipLength=.25)
        if result.metrics:
            issues = {"shoulder_abduction_degrees_2d": round(
                result.metrics.shoulder_angle, 2),
                "elbow_angle_degrees_2d": round(result.metrics.elbow_angle, 2),
                "torso_lean_degrees_2d": round(result.metrics.torso_lean, 2)}
        return issues
    if exercise == "elbow-bend":
        if result.metrics:
            issues = {"elbow_angle_degrees_2d": round(result.metrics.elbow_angle, 2),
                      "upper_arm_angle_degrees_2d": round(
                          result.metrics.upper_arm_angle, 2)}
        if result.metrics:
            shoulder = pixel(f"{side}_shoulder")
            elbow = pixel(f"{side}_elbow")
            wrist = pixel(f"{side}_wrist")
            other = pixel(f"{'right' if side == 'left' else 'left'}_shoulder")
            if shoulder and elbow and wrist and result.metrics:
                upper_limit = (config.upper_arm_start_limit_degrees
                               if result.state == "STRAIGHT" else
                               config.upper_arm_warning_degrees)
                if result.metrics.upper_arm_angle > upper_limit:
                    # A straight-arm raise needs a cue at the *upper arm* even
                    # though the elbow state machine has not begun bending.
                    upper_length = np.linalg.norm(np.asarray(elbow)-shoulder)
                    target = np.asarray(shoulder)+np.array((0., upper_length))
                    vector = target-elbow
                    if np.linalg.norm(vector) > 12:
                        end = np.asarray(elbow)+vector/max(np.linalg.norm(vector), 1)*min(
                            np.linalg.norm(vector), 90)
                        cv2.arrowedLine(frame, elbow, tuple(np.rint(end).astype(int)),
                                        (20, 20, 20), 10, cv2.LINE_AA, tipLength=.28)
                        cv2.arrowedLine(frame, elbow, tuple(np.rint(end).astype(int)),
                                        ORANGE, 6, cv2.LINE_AA, tipLength=.28)
                        cv2.circle(frame, elbow, 12, ORANGE, 3, cv2.LINE_AA)
                    return issues
                bend_cue = (result.state == "BENDING" or
                            result.feedback.startswith("Short bend"))
                straighten_cue = (result.state == "STRAIGHTENING" or
                                  (result.state == "STRAIGHT" and
                                   result.metrics.elbow_angle <
                                   config.rest_angle_degrees))
                if not (bend_cue or straighten_cue):
                    return issues
                forearm_direction = np.arctan2(wrist[1]-elbow[1],
                                                wrist[0]-elbow[0])
                shoulder_direction = np.arctan2(shoulder[1]-elbow[1],
                                                 shoulder[0]-elbow[0])
                toward_shoulder = ((shoulder_direction-forearm_direction+np.pi)
                                   % (2*np.pi)-np.pi)
                # A straight arm has two equal ways to fold; choose the outside
                # of the body so the arc does not cross the torso.
                if abs(toward_shoulder) > np.deg2rad(165):
                    outward = (-1 if other and shoulder[0] < other[0] else 1)
                    toward_shoulder = -outward
                rotation = np.sign(toward_shoulder)
                if not bend_cue:
                    rotation *= -1
                remaining = (result.metrics.elbow_angle-config.bend_target_degrees
                             if bend_cue else
                             config.return_angle_degrees-result.metrics.elbow_angle)
                sweep = np.deg2rad(min(48, max(20, remaining))) * rotation
                radius = 49
                theta = np.linspace(forearm_direction,
                                    forearm_direction+sweep, 18)
                points = np.array([
                    (elbow[0]+radius*np.cos(value),
                     elbow[1]+radius*np.sin(value)) for value in theta
                ], dtype=np.int32)
                cv2.polylines(frame, [points], False, (20, 20, 20), 10,
                              cv2.LINE_AA)
                cv2.polylines(frame, [points], False, ORANGE, 5, cv2.LINE_AA)
                cv2.arrowedLine(frame, tuple(points[-5]), tuple(points[-1]),
                                ORANGE, 6, cv2.LINE_AA, tipLength=.9)
                cv2.circle(frame, elbow, 12, ORANGE, 3, cv2.LINE_AA)
        return issues
    if "torso" in feedback and ("upright" in feedback or "lean" in feedback):
        shoulder, hip = pixel(f"{side}_shoulder"), pixel(f"{side}_hip")
        if shoulder and hip:
            cv2.line(frame, hip, shoulder, ORANGE, 4, cv2.LINE_AA)
            cv2.line(frame, hip, (hip[0], shoulder[1]), (240, 220, 80), 2, cv2.LINE_AA)
            cv2.circle(frame, shoulder, 10, ORANGE, 3, cv2.LINE_AA)
            angle = (getattr(result, "angles", None) or
                     getattr(result, "measurements", None))
            if angle is not None:
                issues["torso_lean_degrees"] = round(angle.torso, 2)
    knee_warning = ((exercise == "squat" and "go lower" in feedback) or
                    (exercise == "toe-touch" and "knee bend" in feedback) or
                    (exercise == "lunge" and "bend the stepping knee" in feedback))
    if knee_warning:
        knee_side = getattr(result, "lead_side", None) or side
        knee = pixel(f"{knee_side}_knee")
        if knee:
            cv2.circle(frame, knee, 17, ORANGE, 3, cv2.LINE_AA)
            if exercise == "squat" and result.angles:
                issues["knee_angle_degrees"] = round(result.angles.knee, 2)
            elif exercise == "toe-touch" and result.measurements:
                issues["knee_flexion_degrees"] = round(result.measurements.knee_flexion, 2)
    if exercise == "cross-body-stretch" and "shoulder height" in feedback:
        wrist, shoulder = pixel(f"{side}_wrist"), pixel(f"{side}_shoulder")
        if wrist and shoulder:
            cv2.circle(frame, wrist, 14, ORANGE, 3, cv2.LINE_AA)
            cv2.line(frame, wrist, (wrist[0], shoulder[1]), ORANGE, 2, cv2.LINE_AA)
            if result.metrics:
                issues["wrist_height_offset"] = round(result.metrics.wrist_height_offset, 3)
    if (exercise == "toe-touch" and result.measurements and
            result.state == "BOTTOM" and
            result.measurements.toe_distance > config.near_toes_distance_ratio):
        finger, toe = pixel(f"{side}_index"), pixel(f"{side}_foot_index")
        if finger and toe:
            cv2.line(frame, finger, toe, ORANGE, 2, cv2.LINE_AA)
            cv2.circle(frame, finger, 10, ORANGE, 2, cv2.LINE_AA)
            issues["finger_to_toe_leg_lengths"] = round(result.measurements.toe_distance, 3)
    return issues


def draw_abduction_dashboard(frame: np.ndarray, result, config: AbductionConfig) -> np.ndarray:
    """Mirrored live feed plus a fixed, legible start/overhead stick-person example."""
    video_width, canvas_width, canvas_height = 960, 1280, 720
    canvas = np.full((canvas_height, canvas_width, 3), (27, 24, 21), np.uint8)
    canvas[:, :video_width] = fit_to_window(frame, video_width, canvas_height)
    ink, muted, orange = (245, 245, 240), (170, 175, 180), ORANGE

    def label(text: str, x: int, y: int, size: float = .63,
              color: tuple[int, int, int] = ink, thickness: int = 1) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thickness, cv2.LINE_AA)

    label("SHOULDER ABDUCTION", 982, 43, .66, ink, 2)
    label(f"State  {result.state}", 982, 83, .58, muted)
    label(f"Cycles  {result.rep_count}    Full arcs  {result.full_arc_count}",
          982, 116, .52, ink)
    angle = f"{result.metrics.shoulder_angle:.0f}" if result.metrics else "--"
    label(f"Shoulder angle  {angle} / 180 deg", 982, 153, .53, orange, 2)
    if result.metrics:
        label(f"Elbow {result.metrics.elbow_angle:.0f} deg   Torso {result.metrics.torso_lean:.0f} deg",
              982, 178, .48, ink)
    feedback_color = (RED if not result.assessment_valid else
                      orange if result.form_warning else GREEN)
    label("LIVE FEEDBACK", 982, 207, .48, muted, 2)
    for row, line in enumerate(wrap_overlay_line(result.feedback, 280, .50)[:2]):
        label(line, 982, 230+row*22, .50, feedback_color, 2)
    label("ONE-ARM SIDE RAISE", 982, 279, .58, muted, 2)

    arm_side = result.side or "left"
    arm_sign = -1 if arm_side == "left" else 1  # Selfie-style video and panel.

    def stick_person(cx: int, top: bool) -> None:
        body = (130, 170, 175)
        head, neck, hip = (cx, 307), (cx, 357), (cx, 471)
        left_shoulder, right_shoulder = (cx-24, 375), (cx+24, 375)
        cv2.circle(canvas, head, 17, body, 3, cv2.LINE_AA)
        for start, end in ((neck, hip), (left_shoulder, right_shoulder),
                           (hip, (cx-22, 552)), (hip, (cx+22, 552))):
            cv2.line(canvas, start, end, body, 3, cv2.LINE_AA)
        active_shoulder = left_shoulder if arm_sign < 0 else right_shoulder
        passive_shoulder = right_shoulder if arm_sign < 0 else left_shoulder
        cv2.line(canvas, passive_shoulder,
                 (passive_shoulder[0]+(-arm_sign)*7, 470), body, 3, cv2.LINE_AA)
        if top:
            elbow = (active_shoulder[0]+arm_sign*4, 325)
            wrist = (active_shoulder[0]+arm_sign*9, 275)
        else:
            elbow = (active_shoulder[0]+arm_sign*5, 422)
            wrist = (active_shoulder[0]+arm_sign*8, 469)
        cv2.line(canvas, active_shoulder, elbow, orange, 5, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, orange, 5, cv2.LINE_AA)
        cv2.circle(canvas, wrist, 6, orange, -1, cv2.LINE_AA)

    stick_person(1037, False)
    stick_person(1212, True)
    cv2.arrowedLine(canvas, (1091, 414), (1140, 414), orange, 3,
                    cv2.LINE_AA, tipLength=.25)
    label("START", 1001, 592, .57, ink, 2)
    label("OVERHEAD", 1151, 592, .53, ink, 2)
    label("Face camera. Start with arm down.", 982, 619, .42, ink)
    label("Lift it sideways to overhead.", 982, 640, .42, ink)
    label("Then lower it to your side.", 982, 661, .42, ink)
    label(f"Near-overhead cutoff: {config.target_angle_degrees:.0f} deg",
          982, 682, .40, muted)
    label("Q quit   F fullscreen   V replay", 982, 702, .43, muted)
    return canvas


def draw_elbow_dashboard(frame: np.ndarray, result, config: ElbowConfig) -> np.ndarray:
    """A simple angle readout and two-pose example next to the mirrored feed."""
    canvas = np.full((720, 1280, 3), (27, 24, 21), np.uint8)
    canvas[:, :960] = fit_to_window(frame, 960, 720)
    ink, muted = (245, 245, 240), (170, 175, 180)

    def label(text: str, x: int, y: int, size: float = .55,
              color: tuple[int, int, int] = ink, thickness: int = 1) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thickness, cv2.LINE_AA)

    label("ELBOW BEND + RETURN", 981, 44, .64, ink, 2)
    label(f"State  {result.state}", 981, 83, .57, muted)
    label(f"Cycles  {result.rep_count}    Full bends  {result.full_bend_count}",
          981, 116, .52)
    angle = f"{result.metrics.elbow_angle:.0f}" if result.metrics else "--"
    label(f"Elbow angle  {angle} deg", 981, 158, .65, ORANGE, 2)
    label(f"Bend target: {config.bend_target_degrees:.0f} deg or less",
          981, 184, .45, muted)
    if result.metrics:
        label(f"Upper arm  {result.metrics.upper_arm_angle:.0f} deg from vertical",
              981, 211, .45)
    feedback_color = (RED if not result.assessment_valid else
                      ORANGE if result.form_warning else GREEN)
    label("LIVE INSTRUCTION", 981, 248, .48, muted, 2)
    for row, line in enumerate(wrap_overlay_line(result.feedback, 285, .49)[:2]):
        label(line, 981, 273+row*22, .49, feedback_color, 2)
    label("EXAMPLE", 981, 334, .56, muted, 2)

    def figure(cx: int, bent: bool) -> None:
        body = (130, 170, 175)
        head, shoulder, hip = (cx, 362), (cx-23, 420), (cx, 523)
        cv2.circle(canvas, head, 16, body, 3, cv2.LINE_AA)
        for start, end in (((cx, 380), hip), ((cx-23, 420), (cx+23, 420)),
                           (hip, (cx-21, 583)), (hip, (cx+21, 583)),
                           ((cx+23, 420), (cx+29, 512))):
            cv2.line(canvas, start, end, body, 3, cv2.LINE_AA)
        elbow = (cx-29, 468)
        wrist = (cx-62, 428) if bent else (cx-34, 516)
        cv2.line(canvas, shoulder, elbow, ORANGE, 5, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, ORANGE, 5, cv2.LINE_AA)
        cv2.circle(canvas, wrist, 6, ORANGE, -1, cv2.LINE_AA)

    figure(1035, False)
    figure(1213, True)
    cv2.arrowedLine(canvas, (1092, 466), (1140, 466), ORANGE, 3,
                    cv2.LINE_AA, tipLength=.25)
    label("STRAIGHT", 989, 618, .51, ink, 2)
    label("BENT", 1178, 618, .51, ink, 2)
    label("1  Keep upper arm by your side", 981, 651, .42)
    label("2  Bend elbow; then straighten", 981, 674, .42)
    label("Q quit   F fullscreen   V replay", 981, 702, .43, muted)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully local PTAssist exercise analysis")
    parser.add_argument("--exercise", choices=("squat", "cross-body-stretch", "toe-touch",
                                               "lunge", "shoulder-abduction", "elbow-bend"),
                        default="squat")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Analyze a recorded local video instead of webcam")
    parser.add_argument("--config", type=Path, help="Exercise config JSON (default chosen by exercise)")
    parser.add_argument("--model", type=Path, default=ROOT / "models/pose_landmarker_lite.task")
    parser.add_argument("--output", type=Path, help="Session JSON path (default: timestamp in sessions/)")
    parser.add_argument("--preview", type=Path, help="Save last annotated frame")
    parser.add_argument("--trace", type=Path, help="Write every frame's landmarks, angles, and analysis to local JSONL")
    parser.add_argument("--no-review-clips", action="store_true",
                        help="Disable local annotated MP4 clips of completed reps")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "references",
                        help="Local folder for user-approved squat/lunge metric references")
    parser.add_argument("--headless", action="store_true", help="No GUI, useful for smoke tests")
    display_mode = parser.add_mutually_exclusive_group()
    display_mode.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F toggles during use)")
    display_mode.add_argument("--windowed", action="store_true", help="Start resizable windowed (default)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0, help="Print a debug frame every N frames")
    args = parser.parse_args()

    config_name = {"squat": "config.json", "cross-body-stretch": "arm_stretch_config.json",
                   "toe-touch": "toe_touch_config.json", "lunge": "lunge_config.json",
                   "shoulder-abduction": "shoulder_abduction_config.json",
                   "elbow-bend": "elbow_bend_config.json"}[args.exercise]
    config_path = args.config or ROOT / config_name
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if args.exercise == "squat":
        config = Config(**config_data)
        analyzer = SquatAnalyzer(config)
    elif args.exercise == "cross-body-stretch":
        config = ArmStretchConfig(**config_data)
        analyzer = CrossBodyStretchAnalyzer(config)
    elif args.exercise == "toe-touch":
        config = ToeTouchConfig(**config_data)
        analyzer = ToeTouchAnalyzer(config)
    elif args.exercise == "shoulder-abduction":
        config = AbductionConfig(**config_data)
        analyzer = ShoulderAbductionAnalyzer(config)
    elif args.exercise == "elbow-bend":
        config = ElbowConfig(**config_data)
        analyzer = ElbowBendAnalyzer(config)
    else:
        config = LungeConfig(**config_data)
        analyzer = LungeAnalyzer(config)
    coach = (ReferenceCoach(args.exercise, args.reference_dir, ComparisonTolerances(
        config.reference_knee_tolerance_degrees,
        config.reference_torso_tolerance_degrees,
        config.reference_duration_tolerance_fraction,
        getattr(config, "reference_stance_tolerance_fraction", 0.0),
    )) if args.exercise in ("squat", "lunge") else None)
    source = str(args.video) if args.video else args.camera
    capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if not args.video else cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {source}")
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    video_fps = capture.get(cv2.CAP_PROP_FPS) if args.video else 0.0
    if args.video and video_fps <= 0:
        capture.release()
        raise RuntimeError("Video FPS is missing; cannot determine frame timestamps")

    start_wall = datetime.now(timezone.utc)
    start_clock = time.monotonic()
    frame_count = 0
    valid_frames = 0
    preview_frame = None
    last_result = None
    observed_rep_count = 0
    reviewed_rep_count = 0
    active_clip = None
    latest_clip = None
    review_clips = []
    error = None
    trace_handle = None
    fullscreen = args.fullscreen
    windowed_size = ((1280, 720) if args.exercise in
                     ("shoulder-abduction", "elbow-bend") else (960, 720))
    if not args.headless:
        desktop_size = screen_size()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            display_size = desktop_size
        else:
            cv2.resizeWindow(WINDOW_NAME, *windowed_size)
            display_size = window_size(windowed_size)
    try:
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = args.trace.open("w", encoding="utf-8", buffering=1)
            trace_handle.write(json.dumps({"type": "meta", "exercise": args.exercise,
                                           "config": asdict(config),
                                           "started_at_utc": start_wall.isoformat()}) + "\n")
        with PoseEstimator(args.model, config.min_detection_confidence) as estimator:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if args.video:
                        break
                    raise RuntimeError("Webcam opened but stopped returning frames")
                frame_count += 1
                timestamp = ((frame_count - 1) / video_fps if args.video
                             else time.monotonic() - start_clock)
                landmarks = estimator.detect(frame, round(timestamp*1000))
                result = analyzer.update(landmarks, timestamp, frame.shape[1], frame.shape[0])
                valid_frames += result.assessment_valid
                if coach and len(analyzer.reps) > observed_rep_count:
                    coach.observe_rep(analyzer.reps[-1])
                    observed_rep_count = len(analyzer.reps)
                if trace_handle:
                    trace_handle.write(json.dumps(frame_record(
                        frame_count, timestamp, frame.shape[1], frame.shape[0],
                        landmarks, result, args.exercise, config)) + "\n")
                draw_skeleton(frame, landmarks)
                issues = draw_issue_markers(frame, landmarks, result, args.exercise, config)
                if args.exercise in ("shoulder-abduction", "elbow-bend"):
                    frame = cv2.flip(frame, 1)  # Keep text readable when mirrored.
                elapsed = max(time.monotonic()-start_clock, 1e-6)
                flash_on = int(time.monotonic() * 4) % 2 == 0
                if args.exercise not in ("shoulder-abduction", "elbow-bend"):
                    draw_overlay(frame, result, frame_count/elapsed, flash_on, args.exercise,
                                 coach, not args.no_review_clips)
                if not args.no_review_clips:
                    idle_state = ("READY" if args.exercise in
                                  ("cross-body-stretch", "shoulder-abduction") else
                                  "STRAIGHT" if args.exercise == "elbow-bend" else "STANDING")
                    if result.state != idle_state and active_clip is None:
                        active_clip = ReviewClip("highlight_measurements")
                    if active_clip is not None:
                        active_clip.add(frame, timestamp, result.feedback, result.reliability, issues)
                    if len(analyzer.reps) > reviewed_rep_count and active_clip is not None:
                        rep = analyzer.reps[-1]
                        review_path = ROOT/"sessions"/(
                            f"{args.exercise}_{start_wall.strftime('%Y%m%d_%H%M%S')}_rep{len(analyzer.reps)}.mp4")
                        duration = max(0.1, active_clip.timeline[-1]["time_seconds"]-
                                       active_clip.timeline[0]["time_seconds"])
                        clip_fps = min(60.0, max(1.0, (len(active_clip.frames)-1)/duration))
                        latest_clip, manifest = active_clip.save(review_path, clip_fps, rep)
                        rep["review_video"] = str(latest_clip)
                        rep["review_timeline"] = str(manifest)
                        review_clips.append(str(latest_clip))
                        active_clip = None
                        reviewed_rep_count = len(analyzer.reps)
                    elif result.state == idle_state and active_clip is not None:
                        active_clip = None
                last_result = result
                if args.log_every and frame_count % args.log_every == 0:
                    print(json.dumps({"frame": frame_count, "timestamp": round(timestamp, 3),
                                      **asdict(result)}), flush=True)
                if not args.headless:
                    if not fullscreen:
                        display_size = window_size(display_size)
                        windowed_size = display_size
                    dashboard = (draw_abduction_dashboard(frame, result, config)
                                 if args.exercise == "shoulder-abduction" else
                                 draw_elbow_dashboard(frame, result, config)
                                 if args.exercise == "elbow-bend" else frame)
                    display_frame = fit_to_window(dashboard, *display_size)
                    draw_status_border(display_frame, result, flash_on)
                    preview_frame = display_frame
                    cv2.imshow(WINDOW_NAME, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("f"):
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                        )
                        if fullscreen:
                            display_size = desktop_size
                        else:
                            cv2.resizeWindow(WINDOW_NAME, *windowed_size)
                            display_size = window_size(windowed_size)
                    approved_path = handle_reference_key(key, coach)
                    if approved_path:
                        print(f"reference_approved={approved_path}", flush=True)
                    if key == ord("v") and latest_clip is not None:
                        play_review(latest_clip)
                else:
                    preview_frame = (draw_abduction_dashboard(frame, result, config)
                                     if args.exercise == "shoulder-abduction" else
                                     draw_elbow_dashboard(frame, result, config)
                                     if args.exercise == "elbow-bend" else frame)
                    draw_status_border(preview_frame, result, flash_on)
                if args.max_frames and frame_count >= args.max_frames:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if trace_handle:
            trace_handle.close()
        capture.release()
        cv2.destroyAllWindows()
        output = args.output or ROOT / "sessions" / f"session_{start_wall.strftime('%Y%m%d_%H%M%S')}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "started_at_utc": start_wall.isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "frames_processed": frame_count,
            "exercise": args.exercise,
            "valid_assessment_frames": valid_frames, "rep_count": len(analyzer.reps),
            "count_definition": {
                "squat": "completed squat state transitions",
                "cross-body-stretch": "cross-body reach, hold, and return cycles; overall form unassessed",
                "toe-touch": "bend, bottom, return, and stand cycles",
                "lunge": "step out, bend, rise, and step back cycles",
                "shoulder-abduction": "one-arm side raise and return cycles; full_arc_reached marks near-overhead",
                "elbow-bend": "one-arm elbow bend and straighten cycles; full_bend_reached marks the angle target",
            }[args.exercise],
            "reps": analyzer.reps, "config": asdict(config),
            "review_clips": review_clips,
            "trace": str(args.trace) if args.trace else None,
            "reference_dir": str(args.reference_dir) if coach else None,
            "reference_events": coach.events if coach else [],
            "last_feedback": last_result.feedback if last_result else None,
            "error": error,
            "limitations": {
                "squat": "2D side-view squat engineering POC; thresholds are not clinically validated",
                "cross-body-stretch": "Cross-body reach and hold timing only; arm straightness/form not assessed "
                                      "because overlapping arms can distort elbow landmarks. "
                                      "Thresholds are not clinically validated",
                "toe-touch": "2D fingertip-to-toe proximity is not proof of contact or flexibility; "
                             "knee-bend and movement thresholds are not clinically validated",
                "lunge": "2D stepping lunge engineering POC; references are user-approved examples, "
                         "not clinical form standards. Thresholds are not clinically validated",
                "shoulder-abduction": "2D front-view shoulder angle approximates abduction; cannot verify "
                                       "true 3D shoulder rotation or safe range. Thresholds are not clinically validated",
                "elbow-bend": "2D elbow angle can be wrong with occlusion or camera depth; "
                              "bend target is an engineering POC value, not a clinical target",
            }[args.exercise],
        }
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.preview and preview_frame is not None:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.preview), preview_frame):
                raise RuntimeError(f"Could not write preview: {args.preview}")
        valid_percent = 100 * valid_frames / frame_count if frame_count else 0.0
        print(f"session={output}", flush=True)
        count_label = ("movement cycles" if args.exercise in
                       ("cross-body-stretch", "shoulder-abduction", "elbow-bend") else "reps")
        print(f"Processed {frame_count} frames; assessed {valid_frames} ({valid_percent:.1f}%); "
              f"counted {len(analyzer.reps)} {count_label}.", flush=True)
        if frame_count and valid_percent < 50:
            setup_hint = {
                "squat": "your full body is visible from the side",
                "cross-body-stretch": "your shoulders and stretching wrist are visible from the front",
                "toe-touch": "your hand and feet are visible from the side",
                "lunge": "both legs and feet are visible from the side",
                "shoulder-abduction": "both shoulders and hips plus the moving arm are visible from the front",
                "elbow-bend": "one complete upper arm and forearm are visible",
            }[args.exercise]
            print(f"Most frames could not be assessed. Check that {setup_hint} "
                  "and watch the on-screen confidence and feedback.", flush=True)


if __name__ == "__main__":
    main()
