import pytest

from ptassist_squat.geometry import joint_angle, lean_from_vertical
from ptassist_squat.landmarks import Landmark


def test_knee_angle_straight_and_bent_with_non_square_frame():
    knee = Landmark(0.5, 0.5)
    ankle = Landmark(0.5, 0.75)
    straight_hip = Landmark(0.5, 0.25)
    right_angle_hip = Landmark(0.6875, 0.5)  # 120 px to right on 640x480 frame
    assert joint_angle(straight_hip, knee, ankle, 640, 480) == pytest.approx(180)
    assert joint_angle(right_angle_hip, knee, ankle, 640, 480) == pytest.approx(90)


def test_torso_lean_relative_to_vertical():
    hip = Landmark(0.5, 0.6)
    shoulder = Landmark(0.5, 0.35)
    diagonal = Landmark(0.6875, 0.35)
    assert lean_from_vertical(shoulder, hip, 640, 480) == pytest.approx(0)
    assert lean_from_vertical(diagonal, hip, 640, 480) == pytest.approx(45)


def test_coincident_landmarks_rejected():
    at = Landmark(0.5, 0.5)
    with pytest.raises(ValueError):
        joint_angle(at, at, Landmark(0.5, 0.7), 640, 480)
