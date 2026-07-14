"""Local state, feedback, and model utilities for the Streamlit tutor UI."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import joblib

APP_VERSION = "1.0.0"
DATA_PATH = Path("data/tutor_progress.json")
SETTINGS_PATH = Path("data/settings.json")


@dataclass
class Progress:
    sessions: int = 0
    attempted: int = 0
    correct: int = 0
    incorrect: int = 0
    confidence_total: float = 0.0
    best_score: int = 0
    activities: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempted * 100 if self.attempted else 0.0

    @property
    def average_confidence(self) -> float:
        return self.confidence_total / self.attempted * 100 if self.attempted else 0.0


DEFAULT_SETTINGS = {"voice": False, "show_fps": True, "dark_mode": False, "camera": 0, "threshold": 0.65}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_progress() -> Progress:
    values = _read(DATA_PATH, {})
    return Progress(**{name: values.get(name, getattr(Progress(), name)) for name in Progress.__dataclass_fields__})


def save_progress(progress: Progress) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(asdict(progress), indent=2), encoding="utf-8")


def load_settings() -> dict:
    return {**DEFAULT_SETTINGS, **_read(SETTINGS_PATH, {})}


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def available_labels() -> list[str]:
    try:
        encoder = joblib.load(Path("models/label_encoder.pkl"))
        return [str(label) for label in encoder.classes_]
    except (OSError, AttributeError, ValueError):
        return list("ABCDEFGHIKLMNOPRSTUVWY")


def feedback(expected: str, predicted: str, confidence: float) -> str:
    if predicted == expected and confidence >= .65:
        return "Excellent hand pose. Hold it steady, then continue to the next sign."
    if confidence < .50:
        return "Good attempt. Keep your hand centered and hold the pose still for a moment."
    hints = ("Raise your index finger slightly.", "Fold your thumb a little more.", "Keep your fingers closer together.")
    return f"Good attempt. You showed {predicted}; target {expected}. {hints[hash(expected) % len(hints)]}"


def record(progress: Progress, expected: str, predicted: str, confidence: float, score: int = 0) -> bool:
    correct = expected == predicted and confidence >= .65
    progress.attempted += 1; progress.confidence_total += confidence
    progress.correct += int(correct); progress.incorrect += int(not correct); progress.best_score = max(progress.best_score, score)
    progress.activities.insert(0, {"time": datetime.now().strftime("%H:%M"), "target": expected, "prediction": predicted, "correct": correct})
    progress.activities = progress.activities[:8]; save_progress(progress)
    return correct
