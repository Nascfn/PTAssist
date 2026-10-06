"""User-approved local movement references and metric comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class ComparisonTolerances:
    knee_degrees: float
    torso_degrees: float
    duration_fraction: float
    stance_fraction: float = 0.0


class ReferenceCoach:
    """Approval is a user label; comparison does not certify exercise form."""

    def __init__(self, exercise: str, directory: Path,
                 tolerances: ComparisonTolerances) -> None:
        if exercise not in ("squat", "lunge"):
            raise ValueError("References currently support squat and lunge")
        self.exercise = exercise
        self.directory = directory
        self.tolerances = tolerances
        self.last_rep: dict | None = None
        self.last_message = "Complete a rep, then press A to approve it as your reference"
        self.events: list[dict] = []
        self._cache: dict[str, dict | None] = {}
        keys = ("squat",) if exercise == "squat" else ("lunge_left", "lunge_right")
        loaded = [key for key in keys if self._load(key)]
        if loaded:
            self.last_message = f"{', '.join(loaded)} reference loaded; do a rep to compare"

    def _key(self, rep: dict) -> str:
        if self.exercise == "squat":
            return "squat"
        side = rep.get("lead_side")
        if side not in ("left", "right"):
            raise ValueError("Lunge rep needs a left or right lead_side")
        return f"lunge_{side}"

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _load(self, key: str) -> dict | None:
        if key not in self._cache:
            path = self._path(key)
            if not path.exists():
                self._cache[key] = None
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("version") != 1 or data.get("exercise") != self.exercise or data.get("key") != key:
                    raise ValueError(f"Invalid reference file: {path}")
                self._cache[key] = data
        return self._cache[key]

    def _clean_metrics(self, rep: dict) -> dict:
        return {key: value for key, value in rep.items()
                if key not in ("number", "reference_comparison", "approved_reference")}

    def approve_last(self) -> Path:
        if self.last_rep is None:
            raise ValueError("Complete a rep before approving a reference")
        rep = self.last_rep
        key = self._key(rep)
        path = self._path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1, "exercise": self.exercise, "key": key,
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "User-approved movement example; not independent clinical form validation",
            "metrics": self._clean_metrics(rep),
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self._cache[key] = data
        rep["approved_reference"] = True
        rep["reference_comparison"] = None
        self.last_message = f"Rep {rep['number']} approved as {key} reference"
        self.events.append({"type": "reference_approved", "rep_number": rep["number"],
                            "key": key, "path": str(path)})
        return path

    def observe_rep(self, rep: dict) -> dict | None:
        self.last_rep = rep
        key = self._key(rep)
        reference = self._load(key)
        if reference is None:
            self.last_message = f"Rep {rep['number']} complete; press A to approve {key} reference"
            rep["reference_comparison"] = None
            return None
        comparison = self._compare(rep, reference["metrics"])
        rep["reference_comparison"] = comparison
        self.last_message = comparison["message"]
        self.events.append({"type": "rep_compared", "rep_number": rep["number"],
                            "key": key, **comparison})
        return comparison

    def message_for_frame(self, result: object) -> str:
        """Give live reference cues only from a currently reliable pose."""
        if not getattr(result, "assessment_valid", False):
            return "Tracking unreliable; reference comparison paused"
        state = getattr(result, "state", "")
        if self.exercise == "squat":
            reference = self._load("squat")
            angles = getattr(result, "angles", None)
            if reference and angles and state == "BOTTOM":
                target = reference["metrics"]
                if angles.torso > target["maximum_torso_lean"]+self.tolerances.torso_degrees:
                    return "Torso leaning farther than reference"
                if angles.knee > target["minimum_knee_angle"]+self.tolerances.knee_degrees:
                    return "Bend deeper to approach your reference"
                return "Current depth and lean are near reference"
        else:
            lead = getattr(result, "lead_side", None)
            measurements = getattr(result, "measurements", None)
            reference = self._load(f"lunge_{lead}") if lead in ("left", "right") else None
            if lead in ("left", "right") and reference is None:
                return f"No {lead} lead reference yet; finish rep then press A"
            if reference and measurements and state == "BOTTOM":
                target = reference["metrics"]
                if measurements.torso > target["maximum_torso_lean"]+self.tolerances.torso_degrees:
                    return "Torso leaning farther than reference"
                knee = getattr(measurements, f"{lead}_knee")
                if knee > target["minimum_lead_knee_angle"]+self.tolerances.knee_degrees:
                    return "Bend lead knee deeper to approach reference"
                if (measurements.stance_ratio < target["maximum_stance_ratio"]*
                        (1-self.tolerances.stance_fraction)):
                    return "Step appears narrower than reference"
                return "Current lunge measurements are near reference"
        return self.last_message

    def _compare(self, rep: dict, reference: dict) -> dict:
        t = self.tolerances
        differences = []
        if self.exercise == "squat":
            knee_key = "minimum_knee_angle"
        else:
            knee_key = "minimum_lead_knee_angle"
        knee_change = rep[knee_key]-reference[knee_key]
        if knee_change > t.knee_degrees:
            differences.append(f"Knee bent {knee_change:.0f} deg less than reference")
        elif knee_change < -t.knee_degrees:
            differences.append(f"Knee bent {-knee_change:.0f} deg more than reference")
        torso_change = rep["maximum_torso_lean"]-reference["maximum_torso_lean"]
        if torso_change > t.torso_degrees:
            differences.append(f"Torso leaned {torso_change:.0f} deg farther than reference")
        elif torso_change < -t.torso_degrees:
            differences.append(f"Torso leaned {-torso_change:.0f} deg less than reference")
        if self.exercise == "lunge":
            stance_change = rep["maximum_stance_ratio"]-reference["maximum_stance_ratio"]
            if reference["maximum_stance_ratio"] > 0 and abs(stance_change)/reference["maximum_stance_ratio"] > t.stance_fraction:
                direction = "wider" if stance_change > 0 else "narrower"
                differences.append(f"Step was {direction} than reference")
        duration_change = rep["duration_seconds"]-reference["duration_seconds"]
        if reference["duration_seconds"] > 0 and abs(duration_change)/reference["duration_seconds"] > t.duration_fraction:
            direction = "longer" if duration_change > 0 else "shorter"
            differences.append(f"Rep took {abs(duration_change):.1f}s {direction} than reference")
        message = ("Within reference tolerances" if not differences else
                   "; ".join(differences[:2])+
                   (f" (+{len(differences)-2} more)" if len(differences) > 2 else ""))
        return {
            "within_reference_tolerances": not differences,
            "differences": differences,
            "message": message,
            "reference_only_not_form_verdict": True,
        }
