import json

from ptassist_squat.reference import ComparisonTolerances, ReferenceCoach
from ptassist_squat.analysis import Angles, FrameResult
from ptassist_squat.lunge import (LungeAnalyzer, LungeConfig, LungeFrameResult,
                                 LungeMeasurements)

from test_lunge import curve as lunge_curve, pose as lunge_pose


def squat_rep(number, knee=95.0, torso=30.0, duration=2.0):
    return {"number": number, "minimum_knee_angle": knee,
            "minimum_hip_angle": 80.0, "maximum_torso_lean": torso,
            "duration_seconds": duration, "minimum_depth_reached": True,
            "excessive_torso_lean": False, "minimum_landmark_reliability": 0.9}


def lunge_rep(number, side="left", knee=95.0, stance=1.2):
    return {"number": number, "lead_side": side,
            "minimum_lead_knee_angle": knee, "minimum_trail_knee_angle": 100.0,
            "maximum_torso_lean": 20.0, "maximum_stance_ratio": stance,
            "duration_seconds": 2.0, "minimum_landmark_reliability": 0.9}


def test_squat_reference_requires_explicit_approval_and_compares_future_reps(tmp_path):
    coach = ReferenceCoach("squat", tmp_path, ComparisonTolerances(12, 10, 0.4))
    first = squat_rep(1)
    assert coach.observe_rep(first) is None
    assert not (tmp_path / "squat.json").exists()
    coach.approve_last()
    saved = json.loads((tmp_path / "squat.json").read_text())
    assert saved["metrics"]["minimum_knee_angle"] == 95
    assert saved["note"].startswith("User-approved")
    assert first["approved_reference"] is True

    similar = coach.observe_rep(squat_rep(2, knee=101, torso=32))
    assert similar["within_reference_tolerances"]
    shallower = coach.observe_rep(squat_rep(3, knee=123, torso=44))
    assert not shallower["within_reference_tolerances"]
    assert any("Knee bent 28 deg less" in item for item in shallower["differences"])
    assert any("Torso leaned 14 deg farther" in item for item in shallower["differences"])


def test_reference_is_loaded_in_a_new_process(tmp_path):
    first = ReferenceCoach("squat", tmp_path, ComparisonTolerances(12, 10, 0.4))
    first.observe_rep(squat_rep(1))
    first.approve_last()
    second = ReferenceCoach("squat", tmp_path, ComparisonTolerances(12, 10, 0.4))
    assert second.observe_rep(squat_rep(1, knee=96))["within_reference_tolerances"]


def test_lunge_references_are_separate_for_each_lead_leg(tmp_path):
    coach = ReferenceCoach("lunge", tmp_path, ComparisonTolerances(12, 10, 0.4, 0.2))
    coach.observe_rep(lunge_rep(1, "left"))
    coach.approve_last()
    assert coach.observe_rep(lunge_rep(2, "right")) is None
    coach.approve_last()
    assert (tmp_path / "lunge_left.json").exists()
    assert (tmp_path / "lunge_right.json").exists()
    shorter = coach.observe_rep(lunge_rep(3, "right", stance=0.8))
    assert any("Step was narrower" in item for item in shorter["differences"])


def test_live_reference_cues_abstain_when_pose_is_invalid(tmp_path):
    coach = ReferenceCoach("squat", tmp_path, ComparisonTolerances(12, 10, 0.4))
    coach.observe_rep(squat_rep(1))
    coach.approve_last()
    bottom = FrameResult("BOTTOM", 1, "right", 0.9, Angles(120, 90, 28),
                         "Go lower", True)
    assert "Bend deeper" in coach.message_for_frame(bottom)
    invalid = FrameResult("BOTTOM", 1, "right", 0.1, None,
                          "Unable to reliably assess", False)
    assert "paused" in coach.message_for_frame(invalid)


def test_live_lunge_cue_uses_matching_lead_side(tmp_path):
    coach = ReferenceCoach("lunge", tmp_path, ComparisonTolerances(12, 10, 0.4, 0.2))
    coach.observe_rep(lunge_rep(1, "left"))
    coach.approve_last()
    bottom = LungeFrameResult("BOTTOM", 1, "left", "left", 0.9,
                              LungeMeasurements(125, 160, 20, 1.2),
                              "Rise from lunge", True)
    assert "Bend lead knee" in coach.message_for_frame(bottom)


def test_counted_lunge_flows_into_reference_comparison(tmp_path):
    analyzer = LungeAnalyzer(LungeConfig())
    coach = ReferenceCoach("lunge", tmp_path, ComparisonTolerances(12, 10, 0.4, 0.2))
    timestamp = 0.0
    for maximum_bend in (1.0, 0.5):
        for step, bend in lunge_curve(max_bend=maximum_bend):
            analyzer.update(lunge_pose(step, bend), timestamp, 640, 480)
            timestamp += 0.08
        rep = analyzer.reps[-1]
        coach.observe_rep(rep)
        if rep["number"] == 1:
            coach.approve_last()
    assert len(analyzer.reps) == 2
    assert analyzer.reps[1]["reference_comparison"]["within_reference_tolerances"] is False
    assert any("Knee bent" in item for item in
               analyzer.reps[1]["reference_comparison"]["differences"])
