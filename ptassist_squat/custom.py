"""Locally taught, pose-template exercise. No model training or clinical form claim."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import deque
import json
import math
from pathlib import Path

import numpy as np

from .landmarks import Landmarks


JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
STEPS = 32


@dataclass(frozen=True)
class CustomConfig:
    camera_width: int = 640
    camera_height: int = 480
    min_detection_confidence: float = 0.5
    min_joint_confidence: float = 0.65
    frame_margin: float = 0.02
    min_demo_frames: int = 12
    min_demo_seconds: float = 0.8
    max_demo_seconds: float = 15.0
    min_joint_excursion: float = 0.17
    min_motion_excursion: float = 0.23
    tolerance_floor: float = 0.22
    max_missing_seconds: float = 0.6
    min_attempt_seconds: float = 0.6
    max_attempt_seconds: float = 15.0
    start_fraction: float = 0.30
    return_fraction: float = 0.19
    min_completion_fraction: float = 0.55
    match_fraction: float = 0.70
    assessment_settle_seconds: float = 0.7
    assessment_stable_center_torso_lengths: float = 0.14
    assessment_stable_scale_fraction: float = 0.06
    assessment_reference_center_torso_lengths: float = 0.55
    assessment_reference_scale_fraction: float = 0.22
    assessment_motion_center_torso_lengths: float = 0.32
    assessment_motion_scale_fraction: float = 0.15


@dataclass
class PoseSample:
    timestamp: float
    points: dict[str, tuple[float, float]]
    confidence: float
    center: tuple[float, float]
    scale: float


def save_examples(path: Path, name: str, examples: list[list[PoseSample]]) -> None:
    """Persist approved local pose examples so teaching can resume after closing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "name": name,
            "examples": [[asdict(sample) for sample in example] for example in examples]}
    path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


def load_examples(path: Path, name: str) -> list[list[PoseSample]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("name") != name:
        raise ValueError(f"Invalid custom examples file: {path}")
    return [[PoseSample(timestamp=float(sample["timestamp"]),
                        points={joint: tuple(point) for joint, point in sample["points"].items()},
                        confidence=float(sample["confidence"]),
                        center=tuple(sample["center"]), scale=float(sample["scale"]))
             for sample in example] for example in data["examples"]]


def sample_pose(landmarks: Landmarks, timestamp: float, width: int, height: int,
                config: CustomConfig) -> PoseSample | None:
    """Express joints in torso lengths around the hip center, independent of camera size."""
    anchor_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    anchors = [landmarks.get(name) for name in anchor_names]
    if any(p is None or p.confidence < config.min_joint_confidence for p in anchors):
        return None
    if any(not (config.frame_margin <= p.x <= 1-config.frame_margin and
                config.frame_margin <= p.y <= 1-config.frame_margin) for p in anchors):
        return None
    shoulders = np.mean([(p.x*width, p.y*height) for p in anchors[:2]], axis=0)
    hips = np.mean([(p.x*width, p.y*height) for p in anchors[2:]], axis=0)
    scale = float(np.linalg.norm(shoulders-hips))
    if scale < 18:
        return None
    points = {}
    confidences = [p.confidence for p in anchors]
    for name in JOINTS:
        p = landmarks.get(name)
        if (p is not None and p.confidence >= config.min_joint_confidence and
                config.frame_margin <= p.x <= 1-config.frame_margin and
                config.frame_margin <= p.y <= 1-config.frame_margin):
            xy = (np.array((p.x*width, p.y*height))-hips)/scale
            points[name] = (float(xy[0]), float(xy[1]))
            confidences.append(p.confidence)
    return PoseSample(timestamp, points, min(confidences),
                      (float(hips[0]), float(hips[1])), scale)


def _resample(values: np.ndarray, steps: int = STEPS) -> np.ndarray:
    old = np.linspace(0, 1, len(values))
    new = np.linspace(0, 1, steps)
    return np.column_stack([np.interp(new, old, values[:, axis]) for axis in range(2)])


def _joint_series(example: list[PoseSample], name: str) -> np.ndarray | None:
    valid = [(i, sample.points[name]) for i, sample in enumerate(example) if name in sample.points]
    if len(valid) < 0.85*len(example) or len(valid) < 2:
        return None
    indices = np.array([item[0] for item in valid])
    xy = np.array([item[1] for item in valid])
    filled = np.column_stack([np.interp(np.arange(len(example)), indices, xy[:, axis])
                              for axis in range(2)])
    return _resample(filled)


@dataclass
class CustomTemplate:
    name: str
    joints: tuple[str, ...]
    trajectory: dict[str, list[list[float]]]
    tolerances: dict[str, float]
    excursion: float
    demo_count: int
    version: int = 1
    reference_center: tuple[float, float] | None = None
    reference_scale: float | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CustomTemplate":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not data.get("joints"):
            raise ValueError(f"Unsupported custom exercise template: {path}")
        data["joints"] = tuple(data["joints"])
        if data.get("reference_center") is not None:
            data["reference_center"] = tuple(data["reference_center"])
        return cls(**data)


def learn_template(name: str, examples: list[list[PoseSample]],
                   config: CustomConfig) -> CustomTemplate:
    if len(examples) < 2:
        raise ValueError("Approve at least two demonstration reps before learning")
    for example in examples:
        if (len(example) < config.min_demo_frames or
                example[-1].timestamp-example[0].timestamp < config.min_demo_seconds or
                example[-1].timestamp-example[0].timestamp > config.max_demo_seconds):
            raise ValueError("Each demonstration must contain a complete 0.8–15 second movement")
    candidates = {}
    for name_of_joint in JOINTS:
        series = [_joint_series(example, name_of_joint) for example in examples]
        if any(s is None for s in series):
            continue
        stack = np.stack(series)
        baseline = np.median(stack[:, :3, :], axis=(0, 1))
        excursion = np.median(np.max(np.linalg.norm(stack-baseline, axis=2), axis=1))
        if excursion >= config.min_joint_excursion:
            candidates[name_of_joint] = stack
    if len(candidates) < 2:
        raise ValueError("Too little reliably tracked joint movement to learn this exercise")
    trajectory = {}
    tolerances = {}
    excursions = []
    for joint, stack in candidates.items():
        median = np.median(stack, axis=0)
        per_demo_excursion = np.max(np.linalg.norm(stack-stack[:, :1, :], axis=2), axis=1)
        return_error = np.linalg.norm(stack[:, -1, :]-stack[:, 0, :], axis=1)
        if np.any(return_error > np.maximum(0.20, per_demo_excursion*0.35)):
            raise ValueError("Each demo must return to its starting pose for automatic rep detection")
        demo_error = np.linalg.norm(stack-median, axis=2)
        tolerance = max(config.tolerance_floor, float(np.percentile(demo_error, 90))*2+0.06)
        if tolerance > 0.75:
            raise ValueError("Approved demos disagree too much; record more consistent reps")
        trajectory[joint] = np.round(median, 4).tolist()
        tolerances[joint] = round(tolerance, 4)
        excursions.append(float(np.max(np.linalg.norm(median-median[0], axis=1))))
    motion_excursion = float(np.mean(excursions))
    if motion_excursion < config.min_motion_excursion:
        raise ValueError("Demonstrations did not move far enough from their starting pose")
    starting = [sample for example in examples for sample in example[:3]]
    reference_center = tuple(float(v) for v in np.median(
        [sample.center for sample in starting], axis=0))
    reference_scale = float(np.median([sample.scale for sample in starting]))
    return CustomTemplate(name, tuple(candidates), trajectory, tolerances,
                          round(motion_excursion, 4), len(examples),
                          reference_center=reference_center,
                          reference_scale=reference_scale)


def _frame_errors(sample: PoseSample, template: CustomTemplate, phase: int) -> dict[str, float]:
    return {joint: math.dist(sample.points[joint], template.trajectory[joint][phase])
            for joint in template.joints if joint in sample.points}


def _closest_phase(sample: PoseSample, template: CustomTemplate, first: int,
                   last: int) -> tuple[int, dict[str, float]]:
    choices = []
    for phase in range(first, last+1):
        errors = _frame_errors(sample, template, phase)
        if len(errors) == len(template.joints):
            choices.append((sum(errors.values())/len(errors), phase, errors))
    _, phase, errors = min(choices)
    return phase, errors


def compare_sequence(samples: list[PoseSample], template: CustomTemplate,
                     config: CustomConfig) -> dict:
    """DTW aligns varied tempos; report localized mismatch, not a medical verdict."""
    n, m = len(samples), STEPS
    costs = np.full((n+1, m+1), np.inf)
    costs[0, 0] = 0.0
    for i, sample in enumerate(samples, 1):
        for j in range(1, m+1):
            errors = _frame_errors(sample, template, j-1)
            if len(errors) != len(template.joints):
                continue
            local = sum(errors.values())/len(errors)
            costs[i, j] = local+min(costs[i-1, j], costs[i, j-1], costs[i-1, j-1])
    if not np.isfinite(costs[n, m]):
        raise ValueError("Attempt has unreliable required landmarks")
    i, j = n, m
    aligned: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        aligned.append((i-1, j-1))
        choices = (costs[i-1, j-1], costs[i-1, j], costs[i, j-1])
        move = int(np.argmin(choices))
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    aligned.reverse()
    per_joint = {joint: [] for joint in template.joints}
    bad_frames: set[int] = set()
    for frame_index, phase in aligned:
        errors = _frame_errors(samples[frame_index], template, phase)
        for joint, error in errors.items():
            per_joint[joint].append(error)
            if error > template.tolerances[joint]:
                bad_frames.add(frame_index)
    deviations = sorted(({
        "joint": joint.replace("_", " "),
        "mean_error_torso_lengths": round(float(np.mean(errors)), 3),
        "max_error_torso_lengths": round(float(np.max(errors)), 3),
        "tolerance_torso_lengths": template.tolerances[joint],
    } for joint, errors in per_joint.items()),
        key=lambda item: item["mean_error_torso_lengths"], reverse=True)
    match_fraction = 1-len(bad_frames)/n
    return {
        "duration_seconds": round(samples[-1].timestamp-samples[0].timestamp, 3),
        "match_percent": round(100*match_fraction, 1),
        "matches_demonstrations": match_fraction >= config.match_fraction,
        "deviations": [d for d in deviations if d["mean_error_torso_lengths"] >
                       d["tolerance_torso_lengths"]*0.55][:3],
        "mismatch_timestamps": [round(samples[i].timestamp, 3) for i in sorted(bad_frames)],
        "minimum_landmark_confidence": round(min(s.confidence for s in samples), 3),
        "interpretation": "Similarity to user-approved examples, not proof of safe or correct form",
    }


@dataclass
class CustomResult:
    state: str
    assessment_valid: bool
    confidence: float
    feedback: str
    attempt_count: int
    phase: int = 0
    errors: dict[str, float] = field(default_factory=dict)
    expected: dict[str, tuple[float, float]] = field(default_factory=dict)
    sample: PoseSample | None = None
    attempt_started: bool = False
    attempt_finished: bool = False
    form_warning: bool = False


class CustomAnalyzer:
    def __init__(self, template: CustomTemplate, config: CustomConfig):
        self.template = template
        self.config = config
        self.state = "READY"
        self.phase = 0
        self.departure_streak = 0
        self.armed = False
        self.warning_streak = 0
        self.peak_distance = 0.0
        self.last_valid_time: float | None = None
        self.attempt: list[PoseSample] = []
        self.attempts: list[dict] = []
        self.still_samples: deque[PoseSample] = deque()
        self.start_center: tuple[float, float] | None = None
        self.start_scale: float | None = None

    def _reset(self) -> None:
        self.state = "READY"
        self.phase = 0
        self.departure_streak = 0
        self.armed = False
        self.warning_streak = 0
        self.peak_distance = 0.0
        self.attempt = []
        self.still_samples.clear()
        self.start_center = None
        self.start_scale = None

    def _at_taught_spot(self, sample: PoseSample) -> bool:
        if self.template.reference_center is None or self.template.reference_scale is None:
            return True  # Compatible with templates saved before location metadata existed.
        return (math.dist(sample.center, self.template.reference_center) <=
                self.config.assessment_reference_center_torso_lengths*self.template.reference_scale and
                abs(sample.scale-self.template.reference_scale)/self.template.reference_scale <=
                self.config.assessment_reference_scale_fraction)

    def _still_at_start(self, sample: PoseSample, timestamp: float) -> bool:
        self.still_samples.append(sample)
        while (self.still_samples and timestamp-self.still_samples[0].timestamp >
               self.config.assessment_settle_seconds+0.15):
            self.still_samples.popleft()
        if (len(self.still_samples) < 4 or
                timestamp-self.still_samples[0].timestamp <
                self.config.assessment_settle_seconds):
            return False
        center = np.array([item.center for item in self.still_samples])
        scale = np.array([item.scale for item in self.still_samples])
        median_scale = float(np.median(scale))
        return (np.max(np.linalg.norm(center-center.mean(axis=0), axis=1)) <=
                self.config.assessment_stable_center_torso_lengths*median_scale and
                (float(scale.max()-scale.min())/median_scale <=
                 self.config.assessment_stable_scale_fraction))

    def _within_attempt_spot(self, sample: PoseSample) -> bool:
        return (self.start_center is not None and self.start_scale is not None and
                math.dist(sample.center, self.start_center) <=
                self.config.assessment_motion_center_torso_lengths*self.start_scale and
                abs(sample.scale-self.start_scale)/self.start_scale <=
                self.config.assessment_motion_scale_fraction)

    def update(self, landmarks: Landmarks, timestamp: float,
               width: int, height: int) -> CustomResult:
        sample = sample_pose(landmarks, timestamp, width, height, self.config)
        return self.update_sample(sample, timestamp)

    def update_sample(self, sample: PoseSample | None, timestamp: float) -> CustomResult:
        if sample is None or any(joint not in sample.points for joint in self.template.joints):
            if (self.last_valid_time is not None and
                    timestamp-self.last_valid_time > self.config.max_missing_seconds):
                self._reset()
            return CustomResult(self.state, False, sample.confidence if sample else 0.0,
                                "Tracking uncertain — assessment paused", len(self.attempts))
        if (self.last_valid_time is not None and
                timestamp-self.last_valid_time > self.config.max_missing_seconds):
            self._reset()
        self.last_valid_time = timestamp
        if not self._at_taught_spot(sample):
            self._reset()
            return CustomResult("READY", False, sample.confidence,
                                "Return to the taught exercise spot", len(self.attempts),
                                sample=sample)
        if self.state == "MOVING" and not self._within_attempt_spot(sample):
            self._reset()
            return CustomResult("READY", False, sample.confidence,
                                "Walk or camera movement excluded", len(self.attempts),
                                sample=sample)
        base_errors = _frame_errors(sample, self.template, 0)
        distance = sum(base_errors.values())/len(base_errors)
        started = finished = False
        start_threshold = max(0.10, self.template.excursion*self.config.start_fraction)
        return_threshold = max(0.065, self.template.excursion*self.config.return_fraction)
        if self.state == "READY":
            if self.armed and not self._within_attempt_spot(sample):
                self._reset()
                return CustomResult("READY", False, sample.confidence,
                                    "Movement away from exercise spot excluded",
                                    len(self.attempts), sample=sample)
            if distance <= return_threshold:
                if self._still_at_start(sample, timestamp):
                    self.armed = True
                    self.start_center = tuple(np.median(
                        [item.center for item in self.still_samples], axis=0))
                    self.start_scale = float(np.median(
                        [item.scale for item in self.still_samples]))
            else:
                self.still_samples.clear()
            self.departure_streak = (self.departure_streak+1 if self.armed and
                                     distance >= start_threshold else 0)
            if self.departure_streak >= 2:
                self.state = "MOVING"
                self.attempt = [sample]
                self.peak_distance = distance
                started = True
        else:
            self.attempt.append(sample)
            self.peak_distance = max(self.peak_distance, distance)
            duration = timestamp-self.attempt[0].timestamp
            if duration > self.config.max_attempt_seconds:
                self._reset()
                return CustomResult("READY", False, sample.confidence,
                                    "Movement timed out — start again", len(self.attempts), sample=sample)
            if (duration >= self.config.min_attempt_seconds and
                    self.peak_distance >= self.template.excursion*self.config.min_completion_fraction and
                    distance <= return_threshold):
                verdict = compare_sequence(self.attempt, self.template, self.config)
                verdict["number"] = len(self.attempts)+1
                self.attempts.append(verdict)
                finished = True
                self._reset()
                feedback = (f"Attempt {verdict['number']}: {verdict['match_percent']:.0f}% match to demos" if
                            verdict["matches_demonstrations"] else
                            f"Attempt {verdict['number']}: review highlighted differences")
                return CustomResult("READY", True, sample.confidence, feedback,
                                    len(self.attempts), sample=sample, attempt_finished=finished,
                                    form_warning=not verdict["matches_demonstrations"])
        if self.state == "READY":
            return CustomResult("READY", True, sample.confidence,
                                "Start moving from demonstrated pose" if self.armed else
                                "Hold demonstrated start pose to arm detector",
                                len(self.attempts), sample=sample)
        phase, errors = _closest_phase(sample, self.template, self.phase,
                                       min(STEPS-1, self.phase+8))
        self.phase = phase
        wrong = [(joint, value) for joint, value in errors.items()
                 if value > self.template.tolerances[joint]]
        wrong.sort(key=lambda item: item[1]/self.template.tolerances[item[0]], reverse=True)
        self.warning_streak = self.warning_streak+1 if wrong else 0
        warning = self.warning_streak >= 2
        feedback = (f"{wrong[0][0].replace('_', ' ').title()} differs from demo — follow cyan guide" if
                    warning else "Movement follows your demonstrations")
        expected = {joint: tuple(self.template.trajectory[joint][phase])
                    for joint in self.template.joints}
        return CustomResult("MOVING", True, sample.confidence, feedback,
                            len(self.attempts), phase, errors, expected, sample,
                            attempt_started=started, form_warning=warning)
