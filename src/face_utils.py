from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_image_paths(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image_bgr(image_path: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Nie udalo sie odczytac obrazu: {image_path}")
    return image_bgr


def build_face_detector(cascade_path: Path | None = None):
    resolved_cascade_path = cascade_path or (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(resolved_cascade_path))
    if detector.empty():
        raise FileNotFoundError(
            f"Nie udalo sie zaladowac klasyfikatora Haar Cascade: {resolved_cascade_path}"
        )
    return detector


def detect_faces(
    image_bgr: np.ndarray,
    *,
    detector,
    min_face_size: int,
    scale_factor: float,
    min_neighbors: int,
):
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )
    return [tuple(int(value) for value in face) for face in faces]


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int):
    return (
        max(0, min(x1, image_width - 1)),
        max(0, min(y1, image_height - 1)),
        max(1, min(x2, image_width)),
        max(1, min(y2, image_height)),
    )


def _square_bbox(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int):
    width = x2 - x1
    height = y2 - y1
    side = max(width, height)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0

    square_x1 = int(round(center_x - side / 2.0))
    square_y1 = int(round(center_y - side / 2.0))
    square_x2 = square_x1 + side
    square_y2 = square_y1 + side

    if square_x1 < 0:
        square_x2 -= square_x1
        square_x1 = 0
    if square_y1 < 0:
        square_y2 -= square_y1
        square_y1 = 0
    if square_x2 > image_width:
        overflow = square_x2 - image_width
        square_x1 = max(0, square_x1 - overflow)
        square_x2 = image_width
    if square_y2 > image_height:
        overflow = square_y2 - image_height
        square_y1 = max(0, square_y1 - overflow)
        square_y2 = image_height

    return _clamp_bbox(
        square_x1,
        square_y1,
        square_x2,
        square_y2,
        image_width,
        image_height,
    )


def extract_face_crops(
    image_bgr: np.ndarray,
    detections: list[tuple[int, int, int, int]],
    *,
    margin_ratio: float,
    square_crop: bool,
    selection: str,
    max_faces: int | None,
):
    if selection not in {"all", "largest"}:
        raise ValueError("selection musi miec wartosc 'all' albo 'largest'.")

    image_height, image_width = image_bgr.shape[:2]
    ordered_detections = sorted(detections, key=lambda item: item[2] * item[3], reverse=True)
    if selection == "largest":
        ordered_detections = ordered_detections[:1]
    if max_faces is not None and max_faces > 0:
        ordered_detections = ordered_detections[:max_faces]

    crops = []
    for face_index, (x, y, width, height) in enumerate(ordered_detections, start=1):
        margin_x = int(round(width * margin_ratio))
        margin_y = int(round(height * margin_ratio))
        x1 = x - margin_x
        y1 = y - margin_y
        x2 = x + width + margin_x
        y2 = y + height + margin_y

        x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, image_width, image_height)
        if square_crop:
            x1, y1, x2, y2 = _square_bbox(x1, y1, x2, y2, image_width, image_height)

        face_bgr = image_bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            continue

        crops.append(
            {
                "face_index": face_index,
                "detected_bbox_xywh": [x, y, width, height],
                "crop_bbox_xyxy": [x1, y1, x2, y2],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area": int(width * height),
                "image_bgr": face_bgr,
                "image_rgb": cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB),
            }
        )

    return crops


def render_face_boxes(image_bgr: np.ndarray, face_records: list[dict]):
    annotated = image_bgr.copy()
    for record in face_records:
        x1, y1, x2, y2 = record["crop_bbox_xyxy"]
        dx, dy, dw, dh = record["detected_bbox_xywh"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (45, 160, 245), 2)
        cv2.rectangle(annotated, (dx, dy), (dx + dw, dy + dh), (60, 215, 85), 2)
        cv2.putText(
            annotated,
            f"face {record['face_index']}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 160, 245),
            2,
            lineType=cv2.LINE_AA,
        )
    return annotated
