"""Face authentication with OpenCV Haar + LBPH recognizer."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceResult:
    authorized: bool
    label: str
    confidence: float


class FaceAuthenticator:
    """Runs synchronous face checks against a pre-trained LBPH model."""

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(str(SETTINGS.haarcascade_path))
        if self._cascade.empty():
            raise FileNotFoundError(f"Unable to load cascade: {SETTINGS.haarcascade_path}")
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not SETTINGS.face_model_path.exists():
            raise FileNotFoundError(f"LBPH model not found: {SETTINGS.face_model_path}")
        self._recognizer.read(str(SETTINGS.face_model_path))
        self._labels = self._load_labels(SETTINGS.face_labels_path)

    @staticmethod
    def _load_labels(labels_path: Path) -> dict[str, str]:
        if not labels_path.exists():
            return {}
        with labels_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def authenticate(self) -> FaceResult:
        cap = cv2.VideoCapture(SETTINGS.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        try:
            for _ in range(SETTINGS.face_detection_max_frames):
                ok, frame = cap.read()
                if not ok:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._cascade.detectMultiScale(
                    gray,
                    scaleFactor=SETTINGS.face_detect_scale_factor,
                    minNeighbors=SETTINGS.face_detect_min_neighbors,
                )
                for (x, y, w, h) in faces:
                    roi = gray[y : y + h, x : x + w]
                    face_id, confidence = self._recognizer.predict(roi)
                    identity = self._labels.get(str(face_id), f"id:{face_id}")
                    authorized = confidence <= SETTINGS.face_confidence_threshold
                    return FaceResult(authorized=authorized, label=identity, confidence=float(confidence))

            return FaceResult(
                authorized=False,
                label="unknown",
                confidence=SETTINGS.face_unknown_confidence,
            )
        finally:
            cap.release()
