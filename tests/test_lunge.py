from ptassist_squat.landmarks import Landmark
from ptassist_squat.lunge import LungeAnalyzer, LungeConfig, LungeState


def pose(step=0.0, bend=0.0, lead="left", confidence=0.98):
    hip_x, hip_y = 320+65*step+30*bend, 210+40*bend
    shoulder_x, shoulder_y = hip_x+20*bend, hip_y-100
    result = {}
    for side, offset in (("left", 0), ("right", 8)):
        moving = side == lead
        ankle_x = 300+(130*step if moving else 0)+offset
        knee_x = 300+(130*step+70*bend if moving else -20*bend)+offset
        points = {
            "shoulder": (shoulder_x+offset, shoulder_y),
            "hip": (hip_x+offset, hip_y),
            "knee": (knee_x, 330+20*bend),
            "ankle": (ankle_x, 445),
        }
        for name, (x, y) in points.items():
            result[f"{side}_{name}"] = Landmark(x/640, y/480, confidence, confidence)
    return result


def curve(max_bend=1.0, max_step=1.0):
    return ([(0.0, 0.0)]*8 +
            [(max_step*i/12, 0.0) for i in range(1, 13)] +
            [(max_step, max_bend*i/20) for i in range(1, 21)] +
            [(max_step, max_bend)]*6 +
            [(max_step, max_bend*(1-i/20)) for i in range(1, 21)] +
            [(max_step*(1-i/12), 0.0) for i in range(1, 13)] +
            [(0.0, 0.0)]*8)


def run(analyzer, max_bend=1.0, max_step=1.0, lead="left"):
    frames = []
    for index, (step, bend) in enumerate(curve(max_bend, max_step)):
        frames.append(analyzer.update(pose(step, bend, lead), index*0.08, 640, 480))
    return frames


def test_forward_lunge_counts_after_step_bend_rise_and_step_back():
    analyzer = LungeAnalyzer(LungeConfig())
    frames = run(analyzer)
    states = []
    for frame in frames:
        if not states or frame.state != states[-1]:
            states.append(frame.state)
    assert states == [state.value for state in LungeState]+[LungeState.STANDING.value]
    assert frames[-1].rep_count == 1
    assert analyzer.reps[0]["lead_side"] == "left"
    assert analyzer.reps[0]["maximum_stance_ratio"] > 0.8
    assert analyzer.reps[0]["minimum_lead_knee_angle"] < 110


def test_right_lead_is_recorded():
    analyzer = LungeAnalyzer(LungeConfig())
    run(analyzer, lead="right")
    assert len(analyzer.reps) == 1
    assert analyzer.reps[0]["lead_side"] == "right"


def test_shallow_lunge_counts_but_depth_flag_is_false():
    analyzer = LungeAnalyzer(LungeConfig())
    run(analyzer, max_bend=0.5)
    assert len(analyzer.reps) == 1
    assert analyzer.reps[0]["minimum_depth_reached"] is False


def test_narrow_squat_like_motion_is_not_a_lunge():
    analyzer = LungeAnalyzer(LungeConfig())
    run(analyzer, max_step=0.0)
    assert analyzer.reps == []


def test_missing_second_leg_abstains_and_cancels_attempt():
    analyzer = LungeAnalyzer(LungeConfig())
    for index, (step, bend) in enumerate(curve()[:25]):
        analyzer.update(pose(step, bend), index*0.08, 640, 480)
    low = pose(1, 1)
    ankle = low["right_ankle"]
    low["right_ankle"] = Landmark(ankle.x, ankle.y, 0.1, 0.1)
    invalid = analyzer.update(low, 2.0, 640, 480)
    assert not invalid.assessment_valid
    assert invalid.measurements is None
    later = analyzer.update({}, 3.0, 640, 480)
    assert later.rep_count == 0
    assert later.state == LungeState.STANDING.value
