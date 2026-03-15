from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .dataset import build_transforms
from .model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predykcja Real vs AI dla pojedynczego obrazu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    model_name = checkpoint["model_name"]

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transform = build_transforms(image_size=image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    confidence, index = torch.max(probabilities, dim=0)

    print(f"Klasa: {class_names[index.item()]}")
    print(f"Pewność: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
