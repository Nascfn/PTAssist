import math
import json
from dataclasses import replace

import numpy as np
import pytest

from ptassist_squat.custom import (CustomAnalyzer, CustomConfig, CustomResult, CustomTemplate,
                                   compare_sequence, learn_template, load_examples,
                                   sample_pose, save_examples)
from ptassist_squat.custom_app import (AMBER, CHOICE_BUTTONS, canvas_point,
                                      clicked_action, draw_dashboard,
                                      draw_joint_guides, draw_practice_guide,
                                      safe_name, target_stick_points)
from ptassist_squat.landmarks import Landmark
from ptassist_squat.review import ReviewClip
from ptassist_squat.teaching import TeachingCapture
from ptassist_squat.teaching_video import TeachingVideoBuffer, play_teaching_video


def pose(progress: float, wrist_offset: float = 0, confidence: float = 0.95):
    lift = math.sin(math.pi*progress)
    points = {
        "left_shoulder": (.44, .35), "right_shoulder": (.56, .35),
        "left_hip": (.44, .55), "right_hip": (.56, .55),
        "left_elbow": (.37, .45-.09*lift),
        "left_wrist": (.31+wrist_offset*lift, .52-.23*lift),
        "right_elbow": (.63, .45), "right_wrist": (.69, .52),
        "left_knee": (.44, .75), "right_knee": (.56, .75),
        "left_ankle": (.44, .95), "right_ankle": (.56, .95),
    }
    return {name: Landmark(x, y, confidence, confidence)
            for name, (x, y) in points.items()}


def samples(offset=0, start_time=0):
    config = CustomConfig()
    return [sample_pose(pose(i/50, offset), start_time+i*.05, 640, 480, config)
            for i in range(51)]


def template():
    return learn_template("arm-raise", [samples(), samples(start_time=3)], CustomConfig())


def test_learns_active_joints_and_saves(tmp_path):
    learned = template()
    assert "left_wrist" in learned.joints
    assert "left_elbow" in learned.joints
    assert "right_wrist" not in learned.joints
    assert learned.reference_center is not None
    assert learned.reference_scale > 0
    path = tmp_path/"custom_arm-raise.json"
    learned.save(path)
    assert CustomTemplate.load(path) == learned
    examples_path = tmp_path/"approved_examples.json"
    demos = [samples(), samples(start_time=3)]
    save_examples(examples_path, "arm-raise", demos)
    restored = load_examples(examples_path, "arm-raise")
    assert restored == demos
    assert learn_template("arm-raise", restored, CustomConfig()).joints == learned.joints


def test_requires_two_complete_examples():
    with pytest.raises(ValueError, match="at least two"):
        learn_template("arm-raise", [samples()], CustomConfig())
    with pytest.raises(ValueError, match="complete"):
        learn_template("arm-raise", [samples()[:5], samples()], CustomConfig())
    one_way = [sample_pose(pose(i/100), i*.05, 640, 480, CustomConfig())
               for i in range(51)]
    with pytest.raises(ValueError, match="return to its starting pose"):
        learn_template("arm-raise", [one_way, one_way], CustomConfig())


def test_good_and_deviating_attempts_have_localized_results():
    learned = template()
    good = compare_sequence(samples(), learned, CustomConfig())
    bad = compare_sequence(samples(offset=.18), learned, CustomConfig())
    assert good["matches_demonstrations"]
    assert not bad["matches_demonstrations"]
    assert bad["deviations"][0]["joint"] == "left wrist"
    assert bad["mismatch_timestamps"]


def test_automatic_attempt_and_confidence_abstention():
    learned = template()
    analyzer = CustomAnalyzer(learned, CustomConfig())
    results = [analyzer.update(pose(0), i*.05, 640, 480) for i in range(20)]
    results += [analyzer.update(pose(i/50), 1+i*.05, 640, 480)
                for i in range(51)]
    assert len(analyzer.attempts) == 1
    assert sum(result.attempt_finished for result in results) == 1
    assert analyzer.attempts[0]["matches_demonstrations"]
    result = analyzer.update(pose(.5, confidence=.1), 3.0, 640, 480)
    assert not result.assessment_valid
    assert result.errors == {}


def test_detector_requires_the_demonstrated_start_pose_before_arming():
    learned = template()
    analyzer = CustomAnalyzer(learned, CustomConfig())
    for i in range(10):
        result = analyzer.update(pose(.5), i*.05, 640, 480)
    assert result.state == "READY"
    assert not result.attempt_started
    assert not analyzer.attempts


def test_assessment_excludes_walk_and_waits_for_stationary_start():
    learned = template()
    analyzer = CustomAnalyzer(learned, CustomConfig())
    first = sample_pose(pose(0), 0, 640, 480, CustomConfig())
    for i in range(24):
        t = i*.05
        moving = replace(first, timestamp=t,
                         center=(first.center[0]+180-i*4, first.center[1]),
                         scale=first.scale*1.45-i*.9)
        result = analyzer.update_sample(moving, t)
        assert not result.attempt_started
    assert not analyzer.armed
    for i in range(20):
        t = 1.2+i*.05
        result = analyzer.update_sample(replace(first, timestamp=t), t)
    assert analyzer.armed
    moved = replace(first, timestamp=2.25,
                    center=(first.center[0]+first.scale*.38, first.center[1]))
    result = analyzer.update_sample(moved, moved.timestamp)
    assert not result.assessment_valid
    assert not analyzer.armed
    assert not analyzer.attempts
    for i in range(20):
        t = 3+i*.05
        analyzer.update_sample(replace(first, timestamp=t), t)
    for i, item in enumerate(samples()):
        t = 4+i*.05
        analyzer.update_sample(replace(item, timestamp=t), t)
    assert len(analyzer.attempts) == 1


def test_teaching_review_video_keeps_only_candidate_frames(tmp_path, monkeypatch):
    import cv2

    buffer = TeachingVideoBuffer()
    frame = np.zeros((90, 120, 3), np.uint8)
    for i in range(12):
        frame[:, :] = (i*10, 40, 160)
        events = ([{"event": "candidate_started"}] if i == 4 else
                  [{"event": "candidate_completed"}] if i == 9 else [])
        buffer.update(frame, i*.1, events)
    assert len(buffer.candidates) == 1
    video = buffer.candidates[0]
    assert video.frames[0][0] <= .1  # Short lead-in shows the starting pose.
    assert video.frames[-1][0] == .9
    path = video.save(tmp_path/"teaching.mp4")
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(video.frames)
    capture.release()
    shown = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append(image.copy()))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)
    play_teaching_video(video, 1, 1)
    assert len(shown) == 1
    assert np.any(shown[0])


def test_rejects_low_confidence_demo_and_sanitizes_name():
    assert sample_pose(pose(.5, confidence=.1), 0, 640, 480, CustomConfig()) is None
    assert safe_name("My Arm Raise!") == "my-arm-raise"
    with pytest.raises(ValueError):
        safe_name("../")


def test_left_arm_raise_can_be_taught_from_waist_up_view():
    config = CustomConfig()
    examples = []
    for start in (0.0, 3.0):
        example = []
        for i in range(51):
            upper = {name: point for name, point in pose(i/50).items()
                     if not name.endswith(("knee", "ankle"))}
            example.append(sample_pose(upper, start+i*.05, 640, 480, config))
        examples.append(example)
    learned = learn_template("left-arm-raise", examples, config)
    assert "left_wrist" in learned.joints
    assert "left_elbow" in learned.joints


def test_dashboard_and_joint_guides_render():
    learned = template()
    analyzer = CustomAnalyzer(learned, CustomConfig())
    for i in range(20):
        analyzer.update(pose(0), i*.05, 640, 480)
    result = None
    for i in range(28):
        result = analyzer.update(pose(i/50, wrist_offset=.18), 1+i*.05, 640, 480)
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_joint_guides(frame, result, learned)
    assert np.any(frame)
    practice = np.zeros_like(frame)
    draw_practice_guide(practice, result, learned)
    assert np.any(practice)
    target = target_stick_points(result, learned)
    assert tuple(target["left_wrist"]) == tuple(result.expected["left_wrist"])
    actual_x = result.sample.center[0]+result.sample.points["left_wrist"][0]*result.sample.scale
    expected_x = result.sample.center[0]+result.expected["left_wrist"][0]*result.sample.scale
    y = round(result.sample.center[1]+result.expected["left_wrist"][1]*result.sample.scale)
    middle_x = round((actual_x+expected_x)/2)
    assert np.linalg.norm(frame[y, middle_x].astype(float)-np.array(AMBER)) < 30
    dashboard = draw_dashboard(frame, "arm-raise", "ASSESS", "", result,
                               2, False, learned, None, False)
    assert dashboard.shape == (810, 1440, 3)
    assert np.any(dashboard[270:420, 1040:1220])  # sidebar stick person
    invalid = CustomResult("MOVING", False, 0, "Tracking uncertain", 0,
                           sample=result.sample, expected=result.expected)
    blank = np.zeros_like(frame)
    draw_joint_guides(blank, invalid, learned)
    assert not np.any(blank)


def test_resizable_mode_buttons_hit_the_correct_action():
    for action, (x0, y0, x1, y1) in CHOICE_BUTTONS.items():
        x, y = (x0+x1)//2, (y0+y1)//2
        display_x, display_y = round(x*1200/1440), round(y*675/810)
        point = canvas_point(display_x, display_y, 1200, 675)
        assert clicked_action(point, "CHOOSE") == action
    assert canvas_point(2, 2, 1200, 900) is None  # Top letterbox.
    frame = np.zeros((480, 640, 3), np.uint8)
    dashboard = draw_dashboard(frame, "left-arm-raise", "CHOOSE", "Choose a mode",
                               None, 2, False, template(), None, False)
    assert dashboard.shape == (810, 1440, 3)


def test_review_clip_saves_video_and_feedback_timeline(tmp_path):
    import cv2

    clip = ReviewClip()
    frame = np.zeros((120, 160, 3), np.uint8)
    for i in range(5):
        frame[:, :, 1] = 30+i*25
        clip.add(frame, i/10, "Left wrist differs from demo", .94,
                 {"left_wrist": .32})
    video, manifest = clip.save(tmp_path/"attempt.mp4", 10, {"match_percent": 45})
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["timeline"][0]["joint_errors_torso_lengths"]["left_wrist"] == .32


def test_teaching_batch_excludes_walks_and_accepts_two_reps_with_pauses():
    capture = TeachingCapture()
    config = CustomConfig()

    def feed(t, progress=0.0, center=(320.0, 264.0), scale=96.0):
        sample = sample_pose(pose(progress), t, 640, 480, config)
        sample = replace(sample, center=center, scale=scale)
        capture.update(sample, t)

    for i in range(21):  # user presses Space while close to the camera
        feed(i*.05, center=(320.0, 320.0), scale=140.0)
    for i in range(1, 29):  # walking back must not count
        fraction = i/28
        feed(1+i*.05, center=(320.0, 320-56*fraction),
             scale=140-44*fraction)
    for i in range(30):  # pause at the exercise spot to arm
        feed(2.45+i*.05)
    assert capture.state == "READY"
    for i in range(41):
        feed(4+i*.05, progress=i/40)
    for i in range(16):  # a pause between reps
        feed(6.05+i*.05)
    for i in range(41):
        feed(6.9+i*.05, progress=i/40)
    for i in range(12):
        feed(8.95+i*.05)
    for i in range(1, 21):  # walk forward to press Space
        fraction = i/20
        feed(9.55+i*.05, center=(320.0, 264+56*fraction),
             scale=96+44*fraction)
    assert len(capture.candidates) == 2
    assert sum(event["event"] == "candidate_completed" for event in capture.events) == 2
    assert all(0.8 <= rep[-1].timestamp-rep[0].timestamp <= 15
               for rep in capture.candidates)
    learned = learn_template("arm-raise", capture.candidates, config)
    assert "left_wrist" in learned.joints
