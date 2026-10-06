from ptassist_squat.arm_stretch import ArmState, ArmStretchConfig, CrossBodyStretchAnalyzer
from ptassist_squat.landmarks import Landmark


def arm_pose(left_reach=0.0, right_reach=0.0, confidence=0.98, bent=False):
    points = {}
    for side, own_x, other_x, reach in (
        ("left", 0.35, 0.65, left_reach),
        ("right", 0.65, 0.35, right_reach),
    ):
        elbow_x = own_x + (other_x-own_x)*reach/2
        elbow_y = 0.35 + (1-reach)*0.12
        wrist_x = own_x + (other_x-own_x)*reach
        wrist_y = 0.35 + (1-reach)*0.24
        if bent and side == "left":
            elbow_y += 0.14
        for part, x, y in (("shoulder", own_x, 0.35),
                           ("elbow", elbow_x, elbow_y),
                           ("wrist", wrist_x, wrist_y)):
            points[f"{side}_{part}"] = Landmark(x, y, confidence, confidence)
    return points


def stretch_curve(hold_frames=22):
    return [0.0]*6 + [i/15 for i in range(1, 16)] + [1.0]*hold_frames + [1-i/15 for i in range(1, 16)] + [0.0]*6


def run(analyzer, curve, side="left"):
    frames = []
    for index, reach in enumerate(curve):
        pose = arm_pose(left_reach=reach) if side == "left" else arm_pose(right_reach=reach)
        frames.append(analyzer.update(pose, index*0.1, 640, 480))
    return frames


def test_full_cross_body_hold_and_return_counts_once():
    analyzer = CrossBodyStretchAnalyzer(ArmStretchConfig())
    frames = run(analyzer, stretch_curve())
    states = []
    for frame in frames:
        if not states or states[-1] != frame.state:
            states.append(frame.state)
    assert states == [state.value for state in ArmState] + [ArmState.READY.value]
    assert frames[-1].rep_count == 1
    assert analyzer.reps[0]["side"] == "left"
    assert analyzer.reps[0]["hold_seconds"] >= 1.5
    assert analyzer.reps[0]["maximum_cross_body_reach"] >= 0.95
    assert analyzer.reps[0]["arm_straightness"] == "not_assessed"


def test_right_arm_and_short_hold():
    complete = CrossBodyStretchAnalyzer(ArmStretchConfig())
    run(complete, stretch_curve(), side="right")
    assert complete.reps[0]["side"] == "right"
    short = CrossBodyStretchAnalyzer(ArmStretchConfig())
    frames = run(short, stretch_curve(hold_frames=2))
    assert frames[-1].rep_count == 0
    assert any(frame.feedback == "Hold longer before returning" for frame in frames)


def test_unreliable_overlapping_elbow_does_not_block_reach():
    analyzer = CrossBodyStretchAnalyzer(ArmStretchConfig())
    for index, reach in enumerate(stretch_curve()):
        landmarks = arm_pose(left_reach=reach, bent=True)
        elbow = landmarks["left_elbow"]
        landmarks["left_elbow"] = Landmark(elbow.x, elbow.y, 0.05, 0.05)
        result = analyzer.update(landmarks, index*0.1, 640, 480)
    assert result.rep_count == 1
    assert not result.form_warning


def test_wrist_too_low_at_full_reach_warns():
    analyzer = CrossBodyStretchAnalyzer(ArmStretchConfig())
    analyzer.update(arm_pose(), 0, 640, 480)
    result = None
    for index in range(1, 8):
        landmarks = arm_pose(left_reach=1.0)
        wrist = landmarks["left_wrist"]
        landmarks["left_wrist"] = Landmark(wrist.x, 0.65, wrist.visibility, wrist.presence)
        result = analyzer.update(landmarks, index*0.1, 640, 480)
    assert result.assessment_valid
    assert result.form_warning
    assert result.feedback == "Keep your arm near shoulder height"


def test_low_confidence_abstains():
    analyzer = CrossBodyStretchAnalyzer(ArmStretchConfig())
    result = analyzer.update(arm_pose(confidence=0.1), 0, 640, 480)
    assert not result.assessment_valid
    assert result.metrics is None
    assert result.feedback == "Unable to reliably assess"


def test_occluded_wrist_abstains_even_with_visible_shoulder():
    analyzer = CrossBodyStretchAnalyzer(ArmStretchConfig())
    landmarks = arm_pose()
    for side in ("left", "right"):
        wrist = landmarks[f"{side}_wrist"]
        landmarks[f"{side}_wrist"] = Landmark(wrist.x, wrist.y, 0.05, 0.05)
    result = analyzer.update(landmarks, 0, 640, 480)
    assert not result.assessment_valid
    assert result.metrics is None
