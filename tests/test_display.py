from ptassist_squat.analysis import Angles, Config, FrameResult
import cv2
import numpy as np

from ptassist_squat.app import (GREEN, ORANGE, RED, display_color,
                               draw_issue_markers, handle_reference_key, wrap_overlay_line)
from ptassist_squat.landmarks import Landmark
from ptassist_squat.reference import ComparisonTolerances, ReferenceCoach


def test_border_color_reflects_assessment_and_form():
    invalid = FrameResult("STANDING", 0, None, 0.0, None, "Unable to reliably assess", False)
    good = FrameResult("STANDING", 0, "right", 0.9, None, "Ready for squat", True)
    warning = FrameResult("BOTTOM", 0, "right", 0.9, None, "Go lower", True, True)
    assert display_color(invalid, True) == RED
    assert display_color(good, True) == GREEN
    assert display_color(warning, True) == ORANGE
    assert display_color(warning, False) is None


def test_reference_key_only_saves_after_a_completed_rep(tmp_path):
    coach = ReferenceCoach("squat", tmp_path, ComparisonTolerances(12, 10, 0.4))
    assert handle_reference_key(ord("a"), coach) is None
    assert not (tmp_path / "squat.json").exists()
    coach.observe_rep({"number": 1, "minimum_knee_angle": 95,
                       "minimum_hip_angle": 80, "maximum_torso_lean": 30,
                       "duration_seconds": 2.0})
    assert handle_reference_key(ord("x"), coach) is None
    assert handle_reference_key(ord("a"), coach) == tmp_path / "squat.json"
    assert (tmp_path / "squat.json").exists()


def test_reference_feedback_wraps_inside_frame():
    text = "Reference: Torso leaned 14 degrees farther than the approved reference rep"
    lines = wrap_overlay_line(text, 400)
    assert len(lines) > 1
    assert all(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] <= 400
               for line in lines)


def test_joint_highlights_require_valid_pose_and_identify_measurement():
    landmarks = {"left_shoulder": Landmark(.42, .27, .9, .9),
                 "left_hip": Landmark(.50, .50, .9, .9),
                 "left_knee": Landmark(.55, .74, .9, .9)}
    good_frame = np.zeros((480, 640, 3), np.uint8)
    warning = FrameResult("BOTTOM", 0, "left", .9, Angles(125, 100, 49),
                          "Keep your torso more upright", True, True)
    issues = draw_issue_markers(good_frame, landmarks, warning, "squat", Config())
    assert issues == {"torso_lean_degrees": 49}
    assert np.any(good_frame)
    invalid_frame = np.zeros_like(good_frame)
    warning.assessment_valid = False
    assert draw_issue_markers(invalid_frame, landmarks, warning, "squat", Config()) == {}
    assert not np.any(invalid_frame)
