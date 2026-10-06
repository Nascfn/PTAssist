"""Teach a movement from approved examples and review local video of later attempts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .app import ROOT, fit_to_window, screen_size
from .custom import (CustomAnalyzer, CustomConfig, CustomResult, CustomTemplate,
                     PoseSample, learn_template, load_examples, sample_pose,
                     save_examples)
from .pose import PoseEstimator, draw_skeleton
from .review import ReviewClip, play_review
from .teaching import TeachingCapture
from .teaching_video import TeachingVideo, TeachingVideoBuffer, play_teaching_video


WINDOW_NAME = "PTAssist | Teach an exercise"
CANVAS_WIDTH, CANVAS_HEIGHT = 1440, 810
VIDEO_WIDTH = 1000
INK = (246, 243, 236)
MUTED = (169, 173, 180)
TEAL = (230, 207, 65)
GREEN = (111, 217, 91)
AMBER = (79, 178, 255)
RED = (94, 91, 245)
PANEL = (35, 29, 23)
BACKGROUND = (27, 23, 18)
STICK_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)
NEUTRAL_STICK = {
    "left_shoulder": (-0.28, -1.0), "right_shoulder": (0.28, -1.0),
    "left_elbow": (-0.48, -0.48), "right_elbow": (0.48, -0.48),
    "left_wrist": (-0.55, 0.05), "right_wrist": (0.55, 0.05),
    "left_hip": (-0.22, 0.0), "right_hip": (0.22, 0.0),
    "left_knee": (-0.25, 1.0), "right_knee": (0.25, 1.0),
    "left_ankle": (-0.27, 2.0), "right_ankle": (0.27, 2.0),
}
MENU_BUTTON = (1330, 76, 1416, 119)
CHOICE_BUTTONS = {
    "learn": (1023, 245, 1416, 330),
    "do": (1023, 353, 1416, 438),
    "teach": (1023, 461, 1416, 546),
}


def _window_size(fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return fallback
    return (width, height) if width > 0 and height > 0 else fallback


def canvas_point(x: int, y: int, width: int, height: int) -> tuple[int, int] | None:
    """Map a click through the letterboxed, resizable OpenCV display."""
    scale = min(width/CANVAS_WIDTH, height/CANVAS_HEIGHT)
    fitted_width = round(CANVAS_WIDTH*scale)
    fitted_height = round(CANVAS_HEIGHT*scale)
    left = (width-fitted_width)//2
    top = (height-fitted_height)//2
    if not (left <= x < left+fitted_width and top <= y < top+fitted_height):
        return None
    return (min(CANVAS_WIDTH-1, int((x-left)/scale)),
            min(CANVAS_HEIGHT-1, int((y-top)/scale)))


def clicked_action(point: tuple[int, int] | None, mode: str) -> str | None:
    if point is None:
        return None
    x, y = point
    if mode == "CHOOSE":
        for action, (x0, y0, x1, y1) in CHOICE_BUTTONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return action
    elif mode in ("TEACH", "PRACTICE", "ASSESS"):
        x0, y0, x1, y1 = MENU_BUTTON
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "menu"
    return None


def safe_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_ //")
    if not slug or len(slug) > 48:
        raise ValueError("Exercise name must contain letters or numbers and be under 49 characters")
    return slug


def _text(frame: np.ndarray, text: str, x: int, y: int, scale: float = 0.7,
          color: tuple[int, int, int] = INK, thickness: int = 1) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _wrap(text: str, max_chars: int = 42) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current)+len(word)+1 > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = (current+" "+word).strip()
    if current:
        lines.append(current)
    return lines


def target_stick_points(result: CustomResult | None,
                        template: CustomTemplate) -> dict[str, tuple[float, float]]:
    """Use the observed stable joints and learned positions for moving joints."""
    points = dict(NEUTRAL_STICK)
    if result and result.assessment_valid and result.sample:
        points.update(result.sample.points)
    phase = result.phase if result and result.state == "MOVING" else 0
    points.update({joint: tuple(template.trajectory[joint][phase])
                   for joint in template.joints})
    return points


def draw_stick_person(frame: np.ndarray, points: dict[str, tuple[float, float]],
                      center: tuple[float, float], scale: float,
                      color: tuple[int, int, int], thickness: int = 3) -> None:
    """Draw a simple head/torso/arms/legs figure in either video or dashboard space."""
    def xy(name: str) -> tuple[int, int]:
        x, y = points[name]
        return round(center[0]+x*scale), round(center[1]+y*scale)

    for start, end in STICK_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, xy(start), xy(end), color, thickness, cv2.LINE_AA)
    for joint in points:
        if joint in NEUTRAL_STICK:
            cv2.circle(frame, xy(joint), max(2, thickness), color, -1, cv2.LINE_AA)
    if "left_shoulder" in points and "right_shoulder" in points:
        left, right = points["left_shoulder"], points["right_shoulder"]
        head = ((left[0]+right[0])/2, (left[1]+right[1])/2-0.35)
        center_px = (round(center[0]+head[0]*scale),
                     round(center[1]+head[1]*scale))
        cv2.circle(frame, center_px, max(6, round(scale*.20)),
                   color, thickness, cv2.LINE_AA)


def draw_joint_guides(frame: np.ndarray, result: CustomResult,
                      template: CustomTemplate) -> None:
    """Ghost the taught stick person and point one large arrow to the main mismatch."""
    if not result.assessment_valid or result.sample is None:
        return
    sample = result.sample
    target = target_stick_points(result, template)
    ghost = frame.copy()
    draw_stick_person(ghost, target, sample.center, sample.scale, TEAL, 2)
    cv2.addWeighted(ghost, 0.62, frame, 0.38, 0, frame)
    if not result.form_warning or not result.expected:
        return
    ranked = sorted(result.errors, key=lambda name:
                    result.errors[name]/template.tolerances[name], reverse=True)
    joint = next((name for name in ranked
                  if result.errors[name] > template.tolerances[name]), None)
    if joint is None:
        return
    actual = sample.points[joint]
    expected = result.expected[joint]
    a = (round(sample.center[0]+actual[0]*sample.scale),
         round(sample.center[1]+actual[1]*sample.scale))
    e = (round(sample.center[0]+expected[0]*sample.scale),
         round(sample.center[1]+expected[1]*sample.scale))
    cv2.arrowedLine(frame, a, e, (21, 20, 18), 10, cv2.LINE_AA, tipLength=.32)
    cv2.arrowedLine(frame, a, e, AMBER, 5, cv2.LINE_AA, tipLength=.32)
    cv2.circle(frame, a, 13, RED, 4, cv2.LINE_AA)
    cv2.circle(frame, e, 12, TEAL, 3, cv2.LINE_AA)


def practice_apex(template: CustomTemplate) -> tuple[str, int] | None:
    wrist_joints = [joint for joint in template.joints if joint.endswith("_wrist")]
    choices = wrist_joints or list(template.joints)
    if not choices:
        return None
    def excursion(joint: str) -> float:
        path = np.asarray(template.trajectory[joint])
        return float(np.max(np.linalg.norm(path-path[0], axis=1)))
    joint = max(choices, key=excursion)
    path = np.asarray(template.trajectory[joint])
    return joint, int(np.argmax(np.linalg.norm(path-path[0], axis=1)))


def draw_practice_guide(frame: np.ndarray, result: CustomResult,
                        template: CustomTemplate) -> None:
    """Show the demonstrated movement's farthest pose and a wrist target arrow."""
    if not result.assessment_valid or result.sample is None:
        return
    apex = practice_apex(template)
    if apex is None:
        return
    joint, phase = apex
    if joint not in result.sample.points:
        return
    path = np.asarray(template.trajectory[joint])
    target = dict(result.sample.points)
    target.update({name: tuple(template.trajectory[name][phase])
                   for name in template.joints})
    sample = result.sample
    ghost = frame.copy()
    draw_stick_person(ghost, target, sample.center, sample.scale, TEAL, 3)
    cv2.addWeighted(ghost, 0.42, frame, 0.58, 0, frame)
    actual = sample.points[joint]
    expected = tuple(path[phase])
    a = (round(sample.center[0]+actual[0]*sample.scale),
         round(sample.center[1]+actual[1]*sample.scale))
    e = (round(sample.center[0]+expected[0]*sample.scale),
         round(sample.center[1]+expected[1]*sample.scale))
    cv2.circle(frame, e, 14, TEAL, 3, cv2.LINE_AA)
    if np.linalg.norm(np.asarray(a)-e) > 25:
        cv2.arrowedLine(frame, a, e, (20, 23, 25), 11, cv2.LINE_AA, tipLength=.28)
        cv2.arrowedLine(frame, a, e, AMBER, 6, cv2.LINE_AA, tipLength=.28)


def draw_review_frame(frame: np.ndarray, result: CustomResult) -> np.ndarray:
    """Burn the relevant feedback into each saved video frame."""
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 54), PANEL, -1)
    color = RED if not result.assessment_valid else AMBER if result.form_warning else GREEN
    _text(annotated, result.feedback[:75], 13, 34, 0.62, color, 2)
    return annotated


def draw_dashboard(frame: np.ndarray, name: str, mode: str, message: str,
                   result: CustomResult | None, examples: int, pending: bool,
                   template: CustomTemplate | None, last_attempt: dict | None,
                   recording: bool, reviewed: bool = False,
                   mirrored: bool = True) -> np.ndarray:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND, np.uint8)
    camera = fit_to_window(frame, VIDEO_WIDTH, CANVAS_HEIGHT-62)
    canvas[62:, :VIDEO_WIDTH] = camera
    cv2.rectangle(canvas, (VIDEO_WIDTH, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), PANEL, -1)
    _text(canvas, "PTASSIST  /  MOVEMENT LAB", 28, 39, 0.75, TEAL, 2)
    _text(canvas, name.upper()[:30], VIDEO_WIDTH+28, 48, 0.92, INK, 2)
    cv2.line(canvas, (VIDEO_WIDTH+27, 70), (CANVAS_WIDTH-27, 70), (69, 66, 61), 1)
    mode_color = TEAL if mode in ("TEACH", "CHOOSE", "PRACTICE") else GREEN
    label = {"CHOOSE": "CHOOSE A MODE", "PRACTICE": "LEARN EXERCISE",
             "ASSESS": "DO EXERCISE", "TEACH": "TEACH COMPUTER"}[mode]
    _text(canvas, label, VIDEO_WIDTH+28, 112, 0.67, mode_color, 2)
    if mode != "CHOOSE" and template:
        cv2.rectangle(canvas, MENU_BUTTON[:2], MENU_BUTTON[2:], (62, 58, 51), -1)
        _text(canvas, "MENU", MENU_BUTTON[0]+10, MENU_BUTTON[1]+29, .58, INK, 2)
    if mode == "TEACH":
        _text(canvas, f"Approved examples  {examples} / 2 minimum",
              VIDEO_WIDTH+28, 151, 0.65)
        if recording:
            _text(canvas, "FINDING REPS - WALKING EXCLUDED", VIDEO_WIDTH+28, 187,
                  0.63, RED, 2)
        elif pending:
            _text(canvas, "REVIEWED: A approve / X discard" if reviewed else
                  "V replay candidate before approval", VIDEO_WIDTH+28,
                  187, 0.55, GREEN if reviewed else AMBER)
        else:
            _text(canvas, "Ready to record an example", VIDEO_WIDTH+28,
                  187, 0.59, MUTED)
        steps = ("1   SPACE  start / stop a batch",
                 "2   Walk back, pause, do 2 reps",
                 "3   Walk forward, SPACE to stop",
                 "4   V replay, then A approve / X discard",
                 "5   L  learn approved reps")
    elif mode == "CHOOSE":
        _text(canvas, "Select how to use your saved demo", VIDEO_WIDTH+28, 162,
              .61, MUTED)
        steps = ()
    else:
        _text(canvas, f"Based on {template.demo_count if template else 0} approved reps",
              VIDEO_WIDTH+28, 151, 0.65)
        _text(canvas, f"Attempts  {result.attempt_count if result else 0}",
              VIDEO_WIDTH+28, 187, 0.69)
        steps = ()
    if mode == "TEACH":
        cv2.rectangle(canvas, (VIDEO_WIDTH+20, 215), (CANVAS_WIDTH-20, 418),
                      (46, 39, 33), -1)
        for i, step in enumerate(steps):
            _text(canvas, step, VIDEO_WIDTH+36, 247+i*35, 0.56,
                  INK if i == 0 else MUTED)
    elif mode == "CHOOSE":
        choices = (("learn", "LEARN EXERCISE", "Follow an arrow to your demo pose"),
                   ("do", "DO EXERCISE", "Count and compare complete attempts"),
                   ("teach", "ADD DEMONSTRATIONS", "Record and review more examples"))
        for action, title, subtitle in choices:
            x0, y0, x1, y1 = CHOICE_BUTTONS[action]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (52, 47, 41), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), TEAL if action == "learn" else
                          GREEN if action == "do" else MUTED, 2)
            _text(canvas, title, x0+17, y0+34, .65, INK, 2)
            _text(canvas, subtitle, x0+17, y0+64, .49, MUTED)
        view_hint = ("FRONT VIEW  /  MOVING ARM VISIBLE" if
                     "arm-raise" in name or "side-raise" in name else
                     "MATCH YOUR DEMO CAMERA VIEW")
        _text(canvas, view_hint,
              VIDEO_WIDTH+28, 603, .51, MUTED)
    elif template:
        cv2.rectangle(canvas, (VIDEO_WIDTH+20, 215), (CANVAS_WIDTH-20, 456),
                      (46, 39, 33), -1)
        _text(canvas, "TARGET STICK PERSON", VIDEO_WIDTH+36, 245, 0.62, TEAL, 2)
        target = target_stick_points(result, template)
        if mode == "PRACTICE":
            apex = practice_apex(template)
            if apex is not None:
                target.update({joint: tuple(template.trajectory[joint][apex[1]])
                               for joint in template.joints})
        shown_target = ({joint: (-point[0], point[1])
                         for joint, point in target.items()} if mirrored else target)
        draw_stick_person(canvas, shown_target,
                          (VIDEO_WIDTH+134, 333), 40, TEAL, 3)
        _text(canvas, "CYAN", VIDEO_WIDTH+250, 296, 0.62, TEAL, 2)
        _text(canvas, "target pose", VIDEO_WIDTH+250, 321, 0.53, MUTED)
        _text(canvas, "ORANGE", VIDEO_WIDTH+250, 362, 0.62, AMBER, 2)
        _text(canvas, "move this way", VIDEO_WIDTH+250, 387, 0.53, MUTED)
        _text(canvas, "V replay   |   T reteach", VIDEO_WIDTH+36, 442, 0.57, INK)
    confidence = result.confidence if result else 0.0
    status_color = (RED if result is not None and not result.assessment_valid else
                    TEAL if mode == "TEACH" else
                    AMBER if result is not None and result.form_warning else GREEN)
    status_y = (650 if mode == "CHOOSE" else
                497 if mode in ("ASSESS", "PRACTICE") else 442)
    _text(canvas, "LIVE STATUS", VIDEO_WIDTH+28, status_y, 0.60, MUTED, 2)
    feedback = (result.feedback if result and mode in ("ASSESS", "PRACTICE") else
                "Show shoulders, hips, and moving joints to teach" if result and
                not result.assessment_valid and mode == "TEACH" else message)
    if mode == "PRACTICE" and result and result.assessment_valid and result.state == "READY":
        feedback = "Follow the cyan demo pose, then lower your arm"
    for i, line in enumerate(_wrap(feedback, 38)[:3]):
        _text(canvas, line, VIDEO_WIDTH+28, status_y+37+i*27,
              0.62, status_color, 2)
    _text(canvas, f"Pose confidence   {confidence:.2f}",
          VIDEO_WIDTH+28, 748 if mode == "CHOOSE" else
          635 if mode in ("ASSESS", "PRACTICE") else 577,
          0.62, MUTED)
    if result and result.state == "MOVING" and template:
        _text(canvas, "Movement progress", VIDEO_WIDTH+28, 669, 0.54, MUTED)
        cv2.rectangle(canvas, (VIDEO_WIDTH+28, 681),
                      (CANVAS_WIDTH-30, 691), (70, 65, 57), -1)
        progress = result.phase/31
        cv2.rectangle(canvas, (VIDEO_WIDTH+28, 681),
                      (VIDEO_WIDTH+28+round((CANVAS_WIDTH-VIDEO_WIDTH-58)*progress), 691),
                      TEAL, -1)
    if last_attempt and mode in ("ASSESS", "PRACTICE"):
        _text(canvas, f"Last attempt: {last_attempt['match_percent']:.0f}% demo match",
              VIDEO_WIDTH+28, 718, 0.61, INK)
        deviations = last_attempt.get("deviations") or []
        if deviations:
            _text(canvas, f"Largest difference: {deviations[0]['joint']}",
                  VIDEO_WIDTH+28, 745, 0.54, AMBER)
    _text(canvas, "LOCAL ONLY  /  MIRROR  /  Q QUIT  /  F FULLSCREEN",
          VIDEO_WIDTH+28, 789, 0.52, MUTED)
    if result and mode in ("ASSESS", "PRACTICE"):
        border = status_color
        cv2.rectangle(canvas, (0, 62), (VIDEO_WIDTH-1, CANVAS_HEIGHT-1),
                      border, 5)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach a local custom exercise from approved reps")
    parser.add_argument("--name", required=True, help="Exercise name, e.g. side-arm-raise")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT/"models/pose_landmarker_lite.task")
    parser.add_argument("--config", type=Path, default=ROOT/"custom_config.json")
    parser.add_argument("--reference-dir", type=Path, default=ROOT/"references")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Show the camera's original orientation")
    parser.add_argument("--mode", choices=("choose", "learn", "do", "teach"),
                        help="Open a mode directly; useful for offline video checks")
    args = parser.parse_args()
    name = safe_name(args.name)
    config = CustomConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    template_path = args.reference_dir/f"custom_{name}.json"
    examples_path = args.reference_dir/f"custom_{name}_examples.json"
    template = CustomTemplate.load(template_path) if template_path.exists() else None
    analyzer = CustomAnalyzer(template, config) if template else None
    if args.mode in ("learn", "do") and template is None:
        raise ValueError("Teach and approve examples before starting Learn or Do mode")
    mode = ({"choose": "CHOOSE", "learn": "PRACTICE", "do": "ASSESS",
             "teach": "TEACH"}[args.mode] if args.mode else
            "ASSESS" if args.headless and template else
            "CHOOSE" if template else "TEACH")
    if mode == "CHOOSE" and template is None:
        mode = "TEACH"
    examples = load_examples(examples_path, name)
    recording: TeachingCapture | None = None
    video_buffer: TeachingVideoBuffer | None = None
    pending: list[list[PoseSample]] | None = None
    pending_videos: list[TeachingVideo] | None = None
    pending_reviewed = False
    approved_demo_videos: list[str] = []
    teaching_events: list[dict] = []
    message = ("Choose guided learning or normal exercise" if mode == "CHOOSE" else
               "Follow the cyan target and return to start" if mode == "PRACTICE" else
               "Wait at your taught spot, then do a rep" if mode == "ASSESS" else
               f"Resumed {len(examples)} approved example(s); record another or press L"
               if examples else
               "Press SPACE to start a batch, then walk back and pause")
    source = str(args.video) if args.video else args.camera
    capture = cv2.VideoCapture(source, cv2.CAP_ANY if args.video else cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {source}")
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    fps = capture.get(cv2.CAP_PROP_FPS) if args.video else 0
    if args.video and fps <= 0:
        capture.release()
        raise RuntimeError("Input video has no FPS")
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    count = 0
    last_result: CustomResult | None = None
    clip: ReviewClip | None = None
    latest_clip: Path | None = None
    clips: list[str] = []
    error = None
    fullscreen = args.fullscreen
    windowed_size = (1200, 675)
    clicks: list[tuple[int, int]] = []
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            desktop = screen_size()
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            display_size = desktop
        else:
            cv2.resizeWindow(WINDOW_NAME, *windowed_size)
            display_size = _window_size(windowed_size)
        def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                point = canvas_point(x, y, *display_size)
                if point is not None:
                    clicks.append(point)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        with PoseEstimator(args.model, config.min_detection_confidence) as estimator:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if args.video:
                        break
                    raise RuntimeError("Webcam opened but stopped returning frames")
                count += 1
                timestamp = ((count-1)/fps if args.video else time.monotonic()-clock)
                landmarks = estimator.detect(frame, round(timestamp*1000))
                sample = sample_pose(landmarks, timestamp, frame.shape[1], frame.shape[0], config)
                new_teaching_events: list[dict] = []
                if mode == "TEACH" and recording is not None:
                    event_count = len(recording.events)
                    recording.update(sample, timestamp)
                    new_teaching_events = recording.events[event_count:]
                    message = recording.prompt
                result = (analyzer.update_sample(sample, timestamp)
                          if mode in ("ASSESS", "PRACTICE") else
                          CustomResult(mode, sample is not None,
                                       sample.confidence if sample else 0.0,
                                       message, 0, sample=sample))
                last_result = result
                draw_skeleton(frame, landmarks)
                if mode == "ASSESS" and result and template:
                    draw_joint_guides(frame, result, template)
                elif mode == "PRACTICE" and result and template:
                    draw_practice_guide(frame, result, template)
                display_frame = cv2.flip(frame, 1) if not args.no_mirror else frame
                if mode == "PRACTICE" and result.assessment_valid and template:
                    apex = practice_apex(template)
                    if apex is not None:
                        cv2.rectangle(display_frame, (7, 8), (345, 43), PANEL, -1)
                        _text(display_frame,
                              f"FOLLOW DEMO: {apex[0].replace('_', ' ').upper()}",
                              16, 34, .57, TEAL, 2)
                elif mode == "ASSESS" and result.form_warning:
                    cv2.rectangle(display_frame, (7, 8), (440, 43), PANEL, -1)
                    _text(display_frame, result.feedback[:45].upper(),
                          16, 34, .52, AMBER, 2)
                if mode == "TEACH" and video_buffer is not None:
                    video_buffer.update(display_frame, timestamp, new_teaching_events)
                if mode in ("ASSESS", "PRACTICE") and result:
                    if result.attempt_started:
                        clip = ReviewClip()
                    if clip is not None:
                        clip.add(draw_review_frame(display_frame, result), timestamp, result.feedback,
                                 result.confidence, result.errors if result.assessment_valid else {})
                    if result.attempt_finished and clip is not None:
                        verdict = analyzer.attempts[-1]
                        path = ROOT/"sessions"/f"custom_{name}_{started.strftime('%Y%m%d_%H%M%S')}_attempt{len(analyzer.attempts)}.mp4"
                        duration = max(0.1, clip.timeline[-1]["time_seconds"]-clip.timeline[0]["time_seconds"])
                        clip_fps = min(60.0, max(1.0, (len(clip.frames)-1)/duration))
                        latest_clip, manifest = clip.save(path, clip_fps, verdict)
                        verdict["review_video"] = str(latest_clip)
                        verdict["review_timeline"] = str(manifest)
                        clips.append(str(latest_clip))
                        clip = None
                    elif not result.assessment_valid and result.state == "READY":
                        clip = None
                last_attempt = analyzer.attempts[-1] if analyzer and analyzer.attempts else None
                dashboard = draw_dashboard(display_frame, name, mode, message, result, len(examples),
                                           pending is not None, template, last_attempt,
                                           recording is not None, pending_reviewed,
                                           mirrored=not args.no_mirror)
                if not args.headless:
                    if not fullscreen:
                        display_size = _window_size(display_size)
                        windowed_size = display_size
                    cv2.imshow(WINDOW_NAME, fit_to_window(dashboard, *display_size))
                    key = cv2.waitKey(1) & 0xFF
                    action = clicked_action(clicks.pop(0), mode) if clicks else None
                    if key in (ord("q"), 27):
                        break
                    if key == ord("f"):
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                        if fullscreen:
                            display_size = desktop = screen_size()
                        else:
                            cv2.resizeWindow(WINDOW_NAME, *windowed_size)
                            display_size = _window_size(windowed_size)
                    elif (key == ord("m") or action == "menu") and template:
                        if recording is not None or pending is not None:
                            message = "Finish or discard the current teaching batch first"
                        else:
                            mode = "CHOOSE"
                            clip = None
                            if analyzer:
                                analyzer._reset()
                            message = "Choose guided learning or normal exercise"
                    elif mode == "CHOOSE" and (key in (ord("l"), ord("d")) or
                                                action in ("learn", "do", "teach")):
                        choice = action or ("learn" if key == ord("l") else "do")
                        if choice == "teach":
                            mode = "TEACH"
                            message = "Press SPACE to record another teaching batch"
                        else:
                            mode = "PRACTICE" if choice == "learn" else "ASSESS"
                            analyzer._reset()
                            message = ("Follow the cyan target and return to start" if
                                       mode == "PRACTICE" else
                                       "Wait at your taught spot, then do a rep")
                    elif key == ord("t"):
                        mode, examples, recording, pending = "TEACH", [], None, None
                        video_buffer, pending_videos, pending_reviewed = None, None, False
                        save_examples(examples_path, name, examples)
                        message = "New teaching set; SPACE starts a batch of reps"
                    elif key == ord(" ") and mode == "TEACH":
                        if recording is None:
                            recording, pending = TeachingCapture(), None
                            video_buffer, pending_videos, pending_reviewed = TeachingVideoBuffer(), None, False
                            message = "Walk back and pause; then do two raises"
                        else:
                            pending = recording.candidates or None
                            pending_videos = video_buffer.candidates or None
                            if (pending is not None and
                                    (pending_videos is None or len(pending_videos) != len(pending))):
                                raise RuntimeError("Teaching video and pose candidates lost synchronization")
                            teaching_events.extend(recording.events)
                            recording, video_buffer, pending_reviewed = None, None, False
                            message = (f"Found {len(pending)} rep(s); V replay the first, then A or X"
                                       if pending else "No complete reps found; try pausing before the first raise")
                    elif key == ord("a") and mode == "TEACH" and pending is not None:
                        candidate = pending[0]
                        if not pending_reviewed:
                            message = "Press V to review this candidate before approving"
                        elif (len(candidate) < config.min_demo_frames or
                                candidate[-1].timestamp-candidate[0].timestamp < config.min_demo_seconds):
                            message = "Demo too short; capture a complete movement"
                        elif candidate[-1].timestamp-candidate[0].timestamp > config.max_demo_seconds:
                            message = "Demo too long; keep each rep under 15 seconds"
                        else:
                            video_path = (ROOT/"sessions"/
                                          f"custom_{name}_{started.strftime('%Y%m%d_%H%M%S')}_demo{len(examples)+1}.mp4")
                            pending_videos[0].save(video_path)
                            approved_demo_videos.append(str(video_path))
                            pending_videos.pop(0)
                            examples.append(pending.pop(0))
                            save_examples(examples_path, name, examples)
                            pending_reviewed = False
                            message = (f"Example {len(examples)} approved; V replay next of {len(pending)}"
                                       if pending else
                                       f"Example {len(examples)} approved; " +
                                       ("press L to learn" if len(examples) >= 2 else
                                        "one more needed"))
                            if not pending:
                                pending = None
                    elif key == ord("x") and mode == "TEACH":
                        if pending:
                            pending.pop(0)
                            pending_videos.pop(0)
                        pending_reviewed = False
                        message = (f"Discarded; {len(pending)} candidate(s) left" if pending else
                                   "Candidate discarded; SPACE to record more")
                        if not pending:
                            pending = None
                    elif key == ord("l") and mode == "TEACH":
                        try:
                            new_template = learn_template(name, examples, config)
                        except ValueError as exc:
                            message = str(exc)
                        else:
                            new_template.save(template_path)
                            template = new_template
                            analyzer = CustomAnalyzer(template, config)
                            mode = "CHOOSE"
                            message = f"Learned {len(template.joints)} moving joints from {len(examples)} reps"
                            print(f"custom_reference={template_path}", flush=True)
                    elif key == ord("v") and mode == "TEACH" and pending:
                        play_teaching_video(pending_videos[0],
                                            len(examples)+1,
                                            len(examples)+len(pending))
                        pending_reviewed = True
                        message = "Reviewed: A approve or X discard this candidate"
                    elif key == ord("v") and latest_clip is not None:
                        play_review(latest_clip)
                if args.max_frames and count >= args.max_frames:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()
        output = args.output or ROOT/"sessions"/f"custom_{name}_{started.strftime('%Y%m%d_%H%M%S')}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "exercise": name, "mode": "custom", "started_at_utc": started.isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "frames_processed": count,
            "approved_examples_this_session": len(examples),
            "examples_path": str(examples_path) if examples else None,
            "teaching_events": teaching_events,
            "template": str(template_path) if template else None,
            "attempts": analyzer.attempts if analyzer else [],
            "review_clips": clips, "approved_demo_videos": approved_demo_videos,
            "mirrored_display": not args.no_mirror,
            "config": asdict(config), "error": error,
            "limitation": "Similarity to user-approved examples only; thresholds are engineering POC values",
        }, indent=2)+"\n", encoding="utf-8")
        print(f"session={output} frames={count} attempts={len(analyzer.attempts) if analyzer else 0}",
              flush=True)


if __name__ == "__main__":
    main()
