import math

from ptassist_squat.analysis import Config, SquatAnalyzer, State
from ptassist_squat.landmarks import Landmark


def pose(knee_angle: float, lean: float = 0, confidence: float = 0.98):
    # Construct a side view in pixel space, then normalize it for the generic API.
    radians = math.radians(knee_angle)
    knee_x, knee_y = 320, 315
    hip_x = knee_x + 120*math.sin(radians)
    hip_y = knee_y + 120*math.cos(radians)
    shoulder_x = hip_x + 100*math.tan(math.radians(lean))
    shoulder_y = hip_y - 100
    coords = {
        "nose": (shoulder_x, shoulder_y-45),
        "shoulder": (shoulder_x, shoulder_y), "hip": (hip_x, hip_y),
        "knee": (knee_x, knee_y), "ankle": (knee_x, 445),
        "heel": (knee_x-20, 452), "foot_index": (knee_x+35, 454),
    }
    result = {f"{side}_{name}": Landmark((x+offset)/640, y/480, confidence, confidence)
            for side, offset in (("left", 0), ("right", 8))
            for name, (x, y) in coords.items() if name != "nose"}
    x, y = coords["nose"]
    result["nose"] = Landmark(x/640, y/480, confidence, confidence)
    return result


def run_curve(analyzer, curve, start=0.0, lean=0, confidence=0.98):
    result = None
    for index, angle in enumerate(curve):
        result = analyzer.update(pose(angle, lean, confidence), start+index*0.08, 640, 480)
    return result, start+len(curve)*0.08


def rep_curve(bottom):
    return [175]*8 + [175+(bottom-175)*i/24 for i in range(1, 25)] + [bottom]*6 + [bottom+(175-bottom)*i/24 for i in range(1, 25)] + [175]*8


def test_full_transition_counts_and_records_depth():
    analyzer = SquatAnalyzer(Config())
    states = []
    for index, angle in enumerate(rep_curve(95)):
        result = analyzer.update(pose(angle), index*0.08, 640, 480)
        if not states or states[-1] != result.state:
            states.append(result.state)
    assert states == [state.value for state in State] + [State.STANDING.value]
    assert result.rep_count == 1
    assert result.state == State.STANDING.value
    assert analyzer.reps[0]["minimum_depth_reached"] is True
    assert 90 <= analyzer.reps[0]["minimum_knee_angle"] <= 110
    assert analyzer.reps[0]["duration_seconds"] > 0
    assert analyzer.reps[0]["minimum_landmark_reliability"] > 0.9


def test_shallow_rep_is_counted_but_fails_depth():
    analyzer = SquatAnalyzer(Config())
    result, _ = run_curve(analyzer, rep_curve(130))
    assert result.rep_count == 1
    assert analyzer.reps[0]["minimum_depth_reached"] is False


def test_too_small_dip_never_enters_bottom_or_counts():
    analyzer = SquatAnalyzer(Config())
    result, _ = run_curve(analyzer, rep_curve(151))
    assert result.rep_count == 0


def test_lean_flag_and_live_feedback():
    analyzer = SquatAnalyzer(Config())
    result, _ = run_curve(analyzer, rep_curve(95), lean=50)
    assert result.feedback == "Keep your torso more upright"
    assert result.form_warning
    assert analyzer.reps[0]["excessive_torso_lean"] is True


def test_saved_lean_near_threshold_matches_flag():
    analyzer = SquatAnalyzer(Config())
    run_curve(analyzer, rep_curve(95), lean=44.96)
    assert analyzer.reps[0]["maximum_torso_lean"] < 45.0
    assert analyzer.reps[0]["excessive_torso_lean"] is False


def test_live_lean_warning_clears_after_torso_recovers_but_rep_keeps_peak():
    analyzer = SquatAnalyzer(Config())
    frames = []
    for index, angle in enumerate(rep_curve(95)):
        lean = 50 if 25 <= index <= 40 else 0
        frames.append(analyzer.update(pose(angle, lean), index*0.08, 640, 480))
    assert any(frame.state == State.BOTTOM.value and frame.form_warning for frame in frames)
    assert any(frame.state == State.ASCENDING.value and frame.angles and
               frame.angles.torso < 45 and not frame.form_warning for frame in frames)
    assert analyzer.reps[0]["excessive_torso_lean"] is True


def test_low_confidence_abstains_and_cancels_incomplete_attempt():
    analyzer = SquatAnalyzer(Config())
    _, timestamp = run_curve(analyzer, [175]*8 + [150]*8)
    assert analyzer.state == State.DESCENDING
    result = analyzer.update(pose(100, confidence=0.1), timestamp, 640, 480)
    assert result.assessment_valid is False
    assert result.angles is None
    assert result.feedback == "Unable to reliably assess"
    result = analyzer.update({}, timestamp+1, 640, 480)
    assert result.rep_count == 0
    assert result.state == State.STANDING.value


def test_out_of_frame_abstains():
    analyzer = SquatAnalyzer(Config())
    landmarks = pose(175)
    landmarks["left_heel"] = Landmark(0.4, 1.05)
    landmarks["right_heel"] = Landmark(0.42, 1.05)
    result = analyzer.update(landmarks, 0, 640, 480)
    assert not result.assessment_valid
    assert result.feedback == "Move back so your full body is visible"


def test_reliable_side_is_selected_automatically():
    analyzer = SquatAnalyzer(Config())
    landmarks = pose(175)
    point = landmarks["left_knee"]
    landmarks["left_knee"] = Landmark(point.x, point.y, 0.1, 0.1)
    result = analyzer.update(landmarks, 0, 640, 480)
    assert result.side == "right"
    assert result.assessment_valid


def test_shallow_bottom_warns_but_descent_does_not():
    analyzer = SquatAnalyzer(Config())
    frames = [analyzer.update(pose(angle), index*0.08, 640, 480)
              for index, angle in enumerate(rep_curve(130))]
    assert any(frame.state == State.DESCENDING.value and not frame.form_warning for frame in frames)
    assert any(frame.state == State.BOTTOM.value and frame.form_warning for frame in frames)
    assert any(frame.state == State.ASCENDING.value and frame.form_warning for frame in frames)


def test_lunge_stance_does_not_count_as_squat():
    analyzer = SquatAnalyzer(Config())
    frames = []
    for index, angle in enumerate(rep_curve(95)):
        landmarks = pose(angle)
        for part in ("ankle", "heel", "foot_index"):
            original = landmarks[f"right_{part}"]
            landmarks[f"right_{part}"] = Landmark(original.x+0.25, original.y,
                                                   original.visibility, original.presence)
        frames.append(analyzer.update(landmarks, index*0.08, 640, 480))
    assert analyzer.reps == []
    assert all(not frame.assessment_valid for frame in frames)
    assert any("too far apart" in frame.feedback for frame in frames)


def test_missing_opposite_ankle_abstains():
    analyzer = SquatAnalyzer(Config())
    landmarks = pose(175)
    for part in ("ankle", "heel", "foot_index"):
        point = landmarks[f"right_{part}"]
        landmarks[f"right_{part}"] = Landmark(point.x, point.y, 0.1, 0.1)
    result = analyzer.update(landmarks, 0, 640, 480)
    assert not result.assessment_valid
    assert "distinguish squat from lunge" in result.feedback


def test_reliable_heel_and_toe_can_replace_uncertain_opposite_ankle():
    analyzer = SquatAnalyzer(Config())
    frames = []
    for index, angle in enumerate(rep_curve(95)):
        landmarks = pose(angle)
        ankle = landmarks["right_ankle"]
        landmarks["right_ankle"] = Landmark(ankle.x, ankle.y, 0.53, 0.53)
        frames.append(analyzer.update(landmarks, index*0.08, 640, 480))
    assert frames[-1].rep_count == 1
    assert any(frame.stance_source == "feet" and frame.assessment_valid for frame in frames)


def test_foot_fallback_still_rejects_a_wide_lunge():
    analyzer = SquatAnalyzer(Config())
    landmarks = pose(175)
    ankle = landmarks["right_ankle"]
    landmarks["right_ankle"] = Landmark(ankle.x, ankle.y, 0.53, 0.53)
    for part in ("heel", "foot_index"):
        point = landmarks[f"right_{part}"]
        landmarks[f"right_{part}"] = Landmark(point.x+0.25, point.y)
    result = analyzer.update(landmarks, 0, 640, 480)
    assert not result.assessment_valid
    assert result.stance_source == "feet"
    assert "too far apart" in result.feedback


def test_step_into_lunge_cancels_squat_attempt():
    analyzer = SquatAnalyzer(Config())
    timestamp = 0.0
    for angle in [175]*8 + [150]*8:
        analyzer.update(pose(angle), timestamp, 640, 480)
        timestamp += 0.08
    assert analyzer.state == State.DESCENDING
    landmarks = pose(120)
    ankle = landmarks["right_ankle"]
    landmarks["right_ankle"] = Landmark(ankle.x+0.25, ankle.y)
    result = analyzer.update(landmarks, timestamp, 640, 480)
    assert result.state == State.STANDING.value
    assert result.rep_count == 0
    assert not result.assessment_valid
