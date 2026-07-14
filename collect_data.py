"""Collect validated, normalized single-hand samples for later model training.

Controls: S start, P pause, N change label, ESC exit.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np

from hand_detection import FEATURE_LENGTH, HandDetector, LandmarkNormalizer


class DataValidator:
    """Rejects invalid and near-duplicate normalized hand samples."""
    def __init__(self) -> None:
        self.previous: Optional[np.ndarray] = None

    def valid(self, raw_hands: list[np.ndarray], normalized: Optional[np.ndarray]) -> bool:
        if len(raw_hands) != 1 or normalized is None or normalized.shape != (21, 3):
            return False
        if not np.isfinite(normalized).all() or not np.allclose(normalized[0], 0.0):
            return False
        # Consecutive nearly identical frames add no useful training information.
        if self.previous is not None and np.allclose(normalized, self.previous, atol=0.003):
            return False
        self.previous = normalized.copy()
        return True


class CSVManager:
    """Owns dataset folders and the stable 63-feature CSV schema."""
    def __init__(self, root: Path = Path("dataset")) -> None:
        self.root = root
        self.raw = root / "raw"; self.processed = root / "processed"; self.csv_dir = root / "csv"
        for folder in (self.raw, self.processed, self.csv_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.path = self.csv_dir / "hand_landmarks.csv"
        self._ensure_header()

    @staticmethod
    def header() -> list[str]:
        return (["sample_id", "timestamp", "label"] +
                [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)] + [f"z_{i}" for i in range(21)])

    def _ensure_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as output:
                csv.writer(output).writerow(self.header())

    def next_id(self) -> int:
        with self.path.open(newline="", encoding="utf-8") as source:
            return max(0, sum(1 for _ in csv.reader(source)) - 1) + 1

    def save(self, sample_id: int, label: str, landmarks: np.ndarray) -> None:
        row = [sample_id, datetime.now().isoformat(timespec="milliseconds"), label]
        row.extend(landmarks[:, 0].tolist()); row.extend(landmarks[:, 1].tolist()); row.extend(landmarks[:, 2].tolist())
        if len(row) != 3 + FEATURE_LENGTH:
            raise ValueError("Feature schema length changed; sample was not saved.")
        with self.path.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(row)

    def count(self) -> int:
        return self.next_id() - 1


class DatasetCollector:
    """Coordinates webcam capture, validation, and CSV persistence."""
    def __init__(self, interval_seconds: float = 0.3) -> None:
        self.csv = CSVManager(); self.validator = DataValidator(); self.normalizer = LandmarkNormalizer()
        self.detector = HandDetector(max_hands=2); self.interval = interval_seconds
        self.collecting = False; self.label = "UNLABELED"; self.saved = 0; self.last_saved = 0.0

    def run(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("ERROR: Cannot open webcam."); self.detector.close(); return
        previous = time.perf_counter()
        print("Dataset collector ready. S=start, P=pause, N=label, ESC=exit.")
        try:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("ERROR: Failed to read webcam frame."); break
                frame = cv2.flip(frame, 1); annotated, hands = self.detector.find_hands(frame)
                normalized = self.normalizer.normalize(hands[0]) if len(hands) == 1 else None
                now = time.monotonic()
                if self.collecting and now - self.last_saved >= self.interval and self.validator.valid(hands, normalized):
                    self.csv.save(self.csv.next_id(), self.label, normalized)
                    self.saved += 1; self.last_saved = now
                fps_now = time.perf_counter(); fps = 1 / max(fps_now - previous, 1e-9); previous = fps_now
                self._overlay(annotated, fps, len(hands))
                cv2.imshow("Dataset Collection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                if key in (ord("s"), ord("S")): self.collecting = True
                elif key in (ord("p"), ord("P")): self.collecting = False
                elif key in (ord("n"), ord("N")):
                    value = input("Enter Label: ").strip()
                    if value: self.label = value.upper(); self.validator.previous = None
        finally:
            camera.release(); self.detector.close(); cv2.destroyAllWindows()
            print("Dataset collector closed safely.")

    def _overlay(self, frame: np.ndarray, fps: float, hands: int) -> None:
        status = "COLLECTING" if self.collecting else "PAUSED"
        lines = (f"Label: {self.label}", f"Status: {status}", f"Session samples: {self.saved}",
                 f"Today's samples: {self.csv.count()}", f"Total dataset size: {self.csv.count()}", f"FPS: {fps:.1f}")
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        if hands != 1:
            cv2.putText(frame, "Show exactly one hand to save", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2)


def verify_dataset_collection() -> None:
    """Hardware-free CSV, label, normalization, and duplicate checks."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        manager = CSVManager(Path(directory) / "dataset")
        raw = np.array([[.1+i*.01, .2+i*.02, i*.001] for i in range(21)], dtype=np.float32)
        normalized = LandmarkNormalizer().normalize(raw)
        validator = DataValidator()
        assert validator.valid([raw], normalized) and not validator.valid([raw], normalized)
        manager.save(1, "HELLO", normalized)
        assert manager.path.exists() and manager.count() == 1
        assert len(next(csv.reader(manager.path.open(encoding="utf-8")))) == 66
    print("Dataset collection checks passed.")


if __name__ == "__main__":
    verify_dataset_collection(); DatasetCollector().run()
