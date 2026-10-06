import math

import numpy as np

from ptassist_squat.app import draw_abduction_dashboard, draw_issue_markers
from ptassist_squat.landmarks import Landmark
from ptassist_squat.shoulder_abduction import (
    AbductionConfig, AbductionState, ShoulderAbductionAnalyzer,
)


def pose(angle=0.0, side="left", bend=0.0, lean=0.0, shift=0.0,
         confidence=0.98):
    shoulders = {"left": (230+lean+shift, 170),
                 "right": (410+lean+shift, 170)}
    hips = {"left": (260+shift, 280), "right": (380+shift, 280)}
    points = {}
    for arm in ("left", "right"):
        shoulder = shoulders[arm]
        arm_angle = angle if arm == side else 0.0
        arm_bend = bend if arm == side else 0.0
        outward = -1 if arm == "left" else 1
        def direction(degrees):
            rad = math.radians(degrees)
            return outward*math.sin(rad), math.cos(rad)
        upper = direction(arm_angle)
        lower = direction(arm_angle-arm_bend)
        elbow = (shoulder[0]+78*upper[0], shoulder[1]+78*upper[1])
        wrist = (elbow[0]+75*lower[0], elbow[1]+75*lower[1])
        for part, xy in (("shoulder", shoulder), ("hip", hips[arm]),
                         ("elbow", elbow), ("wrist", wrist)):
            points[f"{arm}_{part}"] = Landmark(xy[0]/640, xy[1]/480,
                                                 confidence, confidence)
    return points


def run(max_angle=175.0, side="left", bend=0.0, lean=0.0):
    analyzer = ShoulderAbductionAnalyzer(AbductionConfig())
    curve = ([0.0]*12 + [max_angle*i/35 for i in range(1, 36)] +
             [max_angle]*8 + [max_angle*(1-i/35) for i in range(1, 36)] +
             [0.0]*10)
    results = []
    for i, angle in enumerate(curve):
        body_lean = lean if angle >= 50 else 0.0
        results.append(analyzer.update(pose(angle, side, bend, body_lean),
                                       i*.1, 640, 480))
    return analyzer, results


def test_near_overhead_raise_and_return_counts_full_arc_on_either_side():
    for side in ("left", "right"):
        analyzer, results = run(side=side)
        states = [result.state for result in results]
        assert AbductionState.RAISING.value in states
        assert AbductionState.TOP.value in states
        assert AbductionState.LOWERING.value in states
        assert len(analyzer.reps) == 1
        assert analyzer.full_arc_count == 1
        assert analyzer.reps[0]["side"] == side
        assert analyzer.reps[0]["full_arc_reached"]
        assert analyzer.reps[0]["maximum_shoulder_abduction_degrees_2d"] >= 165
        assert not analyzer.reps[0]["elbow_bend_flag"]


def test_short_arc_and_bent_elbow_are_distinguished():
    short, _ = run(max_angle=120)
    assert len(short.reps) == 1
    assert short.full_arc_count == 0
    assert not short.reps[0]["full_arc_reached"]
    bent, frames = run(bend=60)
    assert len(bent.reps) == 1
    # A bent wrist path cannot veto an upper arm that reached the target angle.
    assert bent.reps[0]["full_arc_reached"]
    assert bent.reps[0]["elbow_bend_flag"]
    assert any(frame.feedback == "Straighten your elbow" for frame in frames)


def test_torso_lean_warns_and_low_confidence_abstains():
    leaning, frames = run(lean=45)
    assert len(leaning.reps) == 1
    assert leaning.reps[0]["torso_lean_flag"]
    assert any(frame.feedback == "Keep your torso upright" for frame in frames)
    analyzer = ShoulderAbductionAnalyzer(AbductionConfig())
    low = pose(60, confidence=.1)
    result = analyzer.update(low, 0, 640, 480)
    assert not result.assessment_valid
    assert result.metrics is None
    blank = np.zeros((480, 640, 3), np.uint8)
    assert draw_issue_markers(blank, low, result, "shoulder-abduction",
                              analyzer.config) == {}
    assert not np.any(blank)
    side_view = pose()
    right = side_view["right_shoulder"]
    side_view["right_shoulder"] = Landmark(.41, right.y,
                                            right.visibility, right.presence)
    view_result = analyzer.update(side_view, .1, 640, 480)
    assert not view_result.assessment_valid
    assert view_result.feedback == "Face the camera"


def test_walking_during_a_raise_cancels_it():
    analyzer = ShoulderAbductionAnalyzer(AbductionConfig())
    for i in range(12):
        analyzer.update(pose(), i*.1, 640, 480)
    for i in range(12):
        result = analyzer.update(pose(60*i/11), 1.2+i*.1, 640, 480)
    assert result.state == "RAISING"
    moved = analyzer.update(pose(80, shift=70), 2.5, 640, 480)
    assert not moved.assessment_valid
    assert not analyzer.reps


def test_orange_arrow_and_static_side_preview_render():
    analyzer = ShoulderAbductionAnalyzer(AbductionConfig())
    for i in range(12):
        analyzer.update(pose(), i*.1, 640, 480)
    result = analyzer.update(pose(80), 1.2, 640, 480)
    frame = np.zeros((480, 640, 3), np.uint8)
    measurements = draw_issue_markers(frame, pose(80), result,
                                       "shoulder-abduction", analyzer.config)
    assert "shoulder_abduction_degrees_2d" in measurements
    assert np.any(frame)
    dashboard = draw_abduction_dashboard(frame, result, analyzer.config)
    assert dashboard.shape == (720, 1280, 3)
    assert np.any(dashboard[240:530, 980:1260])


def test_angle_cue_clears_at_top_and_points_down_on_return():
    analyzer, results = run()
    top = next(result for result in results if result.state == "TOP")
    returning = next(result for result in results if result.state == "LOWERING")
    blank = np.zeros((480, 640, 3), np.uint8)
    draw_issue_markers(blank, pose(175), top, "shoulder-abduction",
                       analyzer.config)
    assert not np.any(blank)
    down = np.zeros_like(blank)
    draw_issue_markers(down, pose(120), returning, "shoulder-abduction",
                       analyzer.config)
    assert np.any(down)
