import math

from ptassist_squat.landmarks import Landmark
from ptassist_squat.toe_touch import ToeTouchAnalyzer, ToeTouchConfig, ToeTouchState


def pose(lean: float, closest: float = 0.05, knee_shift: float = 0,
         confidence: float = 0.98):
    hip_x, hip_y = 300+knee_shift, 210
    knee_x, knee_y = 300, 320
    ankle_x, ankle_y = 300, 430
    shoulder_x = hip_x+100*math.sin(math.radians(lean))
    shoulder_y = hip_y-100*math.cos(math.radians(lean))
    leg_length = math.hypot(hip_x-knee_x, hip_y-knee_y)+110
    distance = 0.7-(0.7-closest)*min(lean/75, 1)
    points = {
        "shoulder": (shoulder_x, shoulder_y), "hip": (hip_x, hip_y),
        "knee": (knee_x, knee_y), "ankle": (ankle_x, ankle_y),
        "index": (330, 440-distance*leg_length), "foot_index": (330, 440),
    }
    result = {f"{side}_{name}": Landmark((x+offset)/640, y/480,
                                         confidence, confidence)
              for side, offset in (("left", 0), ("right", 8))
              for name, (x, y) in points.items()}
    result["nose"] = Landmark((shoulder_x+4)/640, (shoulder_y-30)/480,
                               confidence, confidence)
    return result


def curve(max_lean=75):
    return ([0]*8 + [max_lean*i/20 for i in range(1, 21)] +
            [max_lean]*6 + [max_lean*(1-i/20) for i in range(1, 21)] + [0]*8)


def run(analyzer, max_lean=75, closest=0.05, bend_knees=False):
    frames = []
    for index, lean in enumerate(curve(max_lean)):
        shift = 80*min(lean/75, 1) if bend_knees else 0
        frames.append(analyzer.update(pose(lean, closest, shift), index*0.08, 640, 480))
    return frames


def test_straight_knee_toe_touch_counts_and_records_closeness():
    analyzer = ToeTouchAnalyzer(ToeTouchConfig())
    frames = run(analyzer)
    states = []
    for frame in frames:
        if not states or frame.state != states[-1]:
            states.append(frame.state)
    assert states == [state.value for state in ToeTouchState]+[ToeTouchState.STANDING.value]
    assert frames[-1].rep_count == 1
    assert analyzer.reps[0]["near_toes_2d"] is True
    assert analyzer.reps[0]["noticeable_knee_bend"] is False
    assert analyzer.reps[0]["closest_toe_distance_leg_lengths"] < 0.15


def test_bent_knees_are_flagged_without_suppressing_count():
    analyzer = ToeTouchAnalyzer(ToeTouchConfig())
    frames = run(analyzer, bend_knees=True)
    assert frames[-1].rep_count == 1
    assert any(frame.form_warning and frame.feedback == "Noticeable knee bend"
               for frame in frames)
    assert analyzer.reps[0]["noticeable_knee_bend"] is True
    assert analyzer.reps[0]["maximum_knee_flexion"] >= 35


def test_far_reach_counts_but_does_not_claim_toe_contact():
    analyzer = ToeTouchAnalyzer(ToeTouchConfig())
    run(analyzer, closest=0.35)
    assert len(analyzer.reps) == 1
    assert analyzer.reps[0]["near_toes_2d"] is False


def test_small_bend_does_not_count():
    analyzer = ToeTouchAnalyzer(ToeTouchConfig())
    run(analyzer, max_lean=35)
    assert analyzer.reps == []


def test_tracking_loss_abstains_and_cancels_attempt():
    analyzer = ToeTouchAnalyzer(ToeTouchConfig())
    frames = curve()
    for index, lean in enumerate(frames[:20]):
        analyzer.update(pose(lean), index*0.08, 640, 480)
    low = analyzer.update(pose(70, confidence=0.1), 1.6, 640, 480)
    assert not low.assessment_valid
    assert low.measurements is None
    result = analyzer.update({}, 2.5, 640, 480)
    assert result.rep_count == 0
    assert result.state == ToeTouchState.STANDING.value
