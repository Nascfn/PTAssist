from ptassist_squat.analysis import Config, SquatAnalyzer
from ptassist_squat.landmarks import Landmark
from ptassist_squat.lunge import LungeAnalyzer, LungeConfig
from ptassist_squat.trace import frame_record
from ptassist_squat.toe_touch import ToeTouchAnalyzer, ToeTouchConfig

from test_analysis import pose
from test_toe_touch import pose as toe_pose
from test_lunge import pose as lunge_pose


def test_trace_keeps_raw_angles_and_shows_why_frame_was_rejected():
    config = Config()
    landmarks = pose(100)
    landmarks["left_heel"] = Landmark(0.4, 1.05)
    landmarks["right_heel"] = Landmark(0.42, 1.05)
    result = SquatAnalyzer(config).update(landmarks, 0.0, 640, 480)
    record = frame_record(1, 0.0, 640, 480, landmarks, result, "squat", config)

    assert record["analysis"]["assessment_valid"] is False
    assert record["analysis"]["angles"] is None
    assert 90 <= record["raw_angles_diagnostic_only"]["knee"] <= 110
    assert any(issue.endswith("heel:outside_frame") for issue in record["landmark_issues"])
    assert "left_knee" in record["landmarks"]


def test_toe_touch_trace_includes_finger_distance_and_visibility_issue():
    config = ToeTouchConfig()
    landmarks = toe_pose(70)
    for side in ("left", "right"):
        finger = landmarks[f"{side}_index"]
        landmarks[f"{side}_index"] = Landmark(finger.x, finger.y, 0.1, 0.1)
    result = ToeTouchAnalyzer(config).update(landmarks, 0.0, 640, 480)
    record = frame_record(1, 0.0, 640, 480, landmarks, result, "toe-touch", config)
    assert record["raw_angles_diagnostic_only"]["finger_to_toe_leg_lengths"] >= 0
    assert any("index:low_confidence" in issue for issue in record["landmark_issues"])


def test_lunge_trace_includes_both_knees_and_stance_inputs():
    config = LungeConfig()
    landmarks = lunge_pose(1.0, 0.8)
    result = LungeAnalyzer(config).update(landmarks, 0.0, 640, 480)
    record = frame_record(1, 0.0, 640, 480, landmarks, result, "lunge", config)
    assert "left_knee" in record["raw_angles_diagnostic_only"]
    assert "right_knee" in record["raw_angles_diagnostic_only"]
    assert record["analysis"]["measurements"]["stance_ratio"] > 0.8
