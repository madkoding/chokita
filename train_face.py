"""Train LBPH model from dataset directory structure: dataset/<label>/*.jpg"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def build_model(dataset_dir: Path, output_model: Path, labels_path: Path) -> None:
    cascade = cv2.CascadeClassifier(str(SETTINGS.haarcascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Cascade not found: {SETTINGS.haarcascade_path}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[str, int] = {}

    for label_id, person_dir in enumerate(sorted(p for p in dataset_dir.iterdir() if p.is_dir())):
        label_map[person_dir.name] = label_id
        for image_path in person_dir.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces = cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                images.append(image[y : y + h, x : x + w])
                labels.append(label_id)
                break

    if not images:
        raise RuntimeError("No faces detected in dataset")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    output_model.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(output_model))

    inverse_labels = {str(v): k for k, v in label_map.items()}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(inverse_labels, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Model saved to %s", output_model)
    LOGGER.info("Labels saved to %s", labels_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LBPH face model")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--output", type=Path, default=SETTINGS.face_model_path, help="Output .yml path")
    parser.add_argument("--labels", type=Path, default=SETTINGS.face_labels_path, help="Labels json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_model(args.dataset, args.output, args.labels)


if __name__ == "__main__":
    main()
