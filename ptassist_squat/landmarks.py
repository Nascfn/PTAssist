"""Pose-provider-independent normalized landmark data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0
    presence: float = 1.0

    @property
    def confidence(self) -> float:
        return min(self.visibility, self.presence)


Landmarks = dict[str, Landmark]
