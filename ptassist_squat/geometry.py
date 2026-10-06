"""2D image-plane measurements; aspect ratio is accounted for."""

import math

import numpy as np

from .landmarks import Landmark


def point(landmark: Landmark, width: int, height: int) -> np.ndarray:
    return np.array((landmark.x * width, landmark.y * height), dtype=float)


def joint_angle(a: Landmark, joint: Landmark, b: Landmark,
                width: int, height: int) -> float:
    first = point(a, width, height) - point(joint, width, height)
    second = point(b, width, height) - point(joint, width, height)
    product = float(np.linalg.norm(first) * np.linalg.norm(second))
    if product < 1e-6:
        raise ValueError("Coincident landmarks cannot define an angle")
    cosine = float(np.clip(np.dot(first, second) / product, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def lean_from_vertical(shoulder: Landmark, hip: Landmark,
                       width: int, height: int) -> float:
    dx = (shoulder.x - hip.x) * width
    dy = (shoulder.y - hip.y) * height
    if math.hypot(dx, dy) < 1e-6:
        raise ValueError("Coincident shoulder and hip")
    return math.degrees(math.atan2(abs(dx), abs(dy)))
