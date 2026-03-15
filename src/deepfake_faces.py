from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from .dataset import build_transforms
from .model import create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wstępna analiza twarzy na obrazie pod kątem deepfake"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--min-face-size", type=int, default=64)
    return parser.parse_args()


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        model_name=checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["class_names"], checkpoint["image_size"]


def detect_faces(image_bgr, min_face_size: int):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )
    return faces


@torch.no_grad()
def classify_face(model, transform, face_rgb, class_names, device: torch.device):
    pil_image = Image.fromarray(face_rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    confidence, index = torch.max(probabilities, dim=0)
    return class_names[index.item()], confidence.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, image_size = load_classifier(args.checkpoint, device)
    transform = build_transforms(image_size=image_size, train=False)

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Nie udało się odczytać obrazu: {args.image}")

    faces = detect_faces(image_bgr=image_bgr, min_face_size=args.min_face_size)
    if len(faces) == 0:
        print("Nie wykryto twarzy.")
        return

    print(f"Wykryto twarze: {len(faces)}")
    for index, (x, y, width, height) in enumerate(faces, start=1):
        face_bgr = image_bgr[y : y + height, x : x + width]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        label, confidence = classify_face(
            model=model,
            transform=transform,
            face_rgb=face_rgb,
            class_names=class_names,
            device=device,
        )
        print(
            f"Twarz {index}: bbox=({x}, {y}, {width}, {height}), "
            f"klasa={label}, pewność={confidence * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
