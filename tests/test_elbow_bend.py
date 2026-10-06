import math

import numpy as np

from ptassist_squat.app import draw_elbow_dashboard, draw_issue_markers
from ptassist_squat.elbow_bend import (ElbowBendAnalyzer, ElbowConfig,
                                     ElbowFrameResult, ElbowMetrics)
from ptassist_squat.landmarks import Landmark


def pose(angle=175, side="right", shift=0, arm_lift=0, confidence=.98):
    result = {}
    for arm, x in (("left", 220), ("right", 420)):
        moving = arm == side
        degrees = angle if moving else 175
        shoulder = (x+shift, 150)
        elbow = (x+shift+(arm_lift if moving else 0), 220)
        turn = math.radians(180-degrees)
        outward = -1 if arm == "left" else 1
        wrist = (elbow[0]+outward*70*math.sin(turn),
                 elbow[1]+70*math.cos(turn))
        for part, xy in (("shoulder", shoulder), ("elbow", elbow),
                         ("wrist", wrist)):
            result[f"{arm}_{part}"] = Landmark(xy[0]/640, xy[1]/480,
                                                  confidence, confidence)
    return result


def cycle(low=65, side="right", arm_lift=0):
    analyzer = ElbowBendAnalyzer(ElbowConfig())
    angles = ([175]*10 + [175-(175-low)*i/25 for i in range(1, 26)] +
              [low]*5 + [low+(175-low)*i/25 for i in range(1, 26)] + [175]*8)
    results = [analyzer.update(pose(angle, side, arm_lift=arm_lift),
                               i*.1, 640, 480)
               for i, angle in enumerate(angles)]
    return analyzer, results


def test_bend_and_return_counts_both_arms():
    for side in ("left", "right"):
        analyzer, results = cycle(side=side)
        assert {"BENDING", "BENT", "STRAIGHTENING"}.issubset(
            {result.state for result in results})
        assert len(analyzer.reps) == 1
        assert analyzer.full_bend_count == 1
        assert analyzer.reps[0]["side"] == side
        assert analyzer.reps[0]["full_bend_reached"]
        assert analyzer.reps[0]["minimum_elbow_angle_degrees_2d"] <= 95


def test_shallow_bend_counted_but_marked_short():
    analyzer, _ = cycle(low=115)
    assert len(analyzer.reps) == 1
    assert analyzer.full_bend_count == 0
    assert not analyzer.reps[0]["full_bend_reached"]


def test_missing_wrist_abstains_and_walk_cancels():
    analyzer = ElbowBendAnalyzer(ElbowConfig())
    low = pose(90, confidence=.1)
    result = analyzer.update(low, 0, 640, 480)
    assert not result.assessment_valid
    blank = np.zeros((480, 640, 3), np.uint8)
    assert draw_issue_markers(blank, low, result, "elbow-bend", analyzer.config) == {}
    assert not np.any(blank)
    for i in range(10):
        analyzer.update(pose(), 1+i*.1, 640, 480)
    for i in range(10):
        result = analyzer.update(pose(175-i*7), 2+i*.1, 640, 480)
    assert result.state == "BENDING"
    moved = analyzer.update(pose(80, shift=100), 3.1, 640, 480)
    assert not moved.assessment_valid
    assert not analyzer.reps


def test_dashboard_and_angle_based_arrow():
    analyzer, results = cycle()
    bending = next(result for result in results if result.state == "BENDING")
    bent = next(result for result in results if result.state == "BENT")
    frame = np.zeros((480, 640, 3), np.uint8)
    measurements = draw_issue_markers(frame, pose(125), bending,
                                       "elbow-bend", analyzer.config)
    assert "elbow_angle_degrees_2d" in measurements
    assert np.any(frame)
    clear = np.zeros_like(frame)
    draw_issue_markers(clear, pose(65), bent, "elbow-bend", analyzer.config)
    assert not np.any(clear)
    dashboard = draw_elbow_dashboard(frame, bending, analyzer.config)
    assert dashboard.shape == (720, 1280, 3)
    assert np.any(dashboard[350:580, 980:1260])


def test_straight_arm_raise_has_visible_upper_arm_correction():
    config = ElbowConfig()
    result = ElbowFrameResult("STRAIGHT", 0, 0, "right", .95,
                              ElbowMetrics(175, 55),
                              "Straighten one arm at your side to start",
                              True, True)
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_issue_markers(frame, pose(175, arm_lift=100), result,
                       "elbow-bend", config)
    assert np.any(frame[180:340, 390:560])


def test_straight_arm_raised_is_not_a_ready_elbow_rep():
    analyzer = ElbowBendAnalyzer(ElbowConfig())
    raised = {
        "right_shoulder": Landmark(420/640, 150/480),
        "right_elbow": Landmark(500/640, 220/480),
        "right_wrist": Landmark(562/640, 260/480),
    }
    for i in range(15):
        result = analyzer.update(raised, i*.1, 640, 480)
    assert result.state == "STRAIGHT"
    assert result.form_warning
    assert result.feedback == "Lower your upper arm to your side"
    assert not analyzer.armed


def test_short_bend_completion_keeps_a_bend_cue_visible():
    config = ElbowConfig()
    result = ElbowFrameResult("STRAIGHT", 1, 0, "right", .95,
                              ElbowMetrics(168, 10), "Short bend recorded",
                              True, True)
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_issue_markers(frame, pose(168), result, "elbow-bend", config)
    assert np.any(frame)
