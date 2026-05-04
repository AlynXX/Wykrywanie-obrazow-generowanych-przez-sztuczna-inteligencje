from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .inference import (
    load_model_bundle,
    predict_image,
    predict_with_grad_cam,
    save_grad_cam_overlay,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predykcja Real vs AI dla pojedynczego obrazu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Opcjonalny prog decyzyjny dla klasy fake w zadaniu binarnym.",
    )
    parser.add_argument(
        "--save-cam",
        type=Path,
        default=None,
        help="Sciezka zapisu obrazu z nalozona mapa Grad-CAM.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Opcjonalna nazwa warstwy dla Grad-CAM. Domyslnie ostatnia Conv2d.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Opcjonalny indeks klasy do wyjasnienia. Domyslnie klasa przewidziana przez model.",
    )
    parser.add_argument(
        "--cam-alpha",
        type=float,
        default=0.45,
        help="Przezroczystosc nakladki Grad-CAM w zakresie 0-1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_model_bundle(args.checkpoint, decision_threshold=args.threshold)
    image = Image.open(args.image).convert("RGB")

    if args.save_cam is not None:
        result = predict_with_grad_cam(
            bundle=bundle,
            image=image,
            target_layer_name=args.target_layer,
            target_class=args.target_class,
            cam_alpha=args.cam_alpha,
        )
        save_grad_cam_overlay(
            image=result["input_image"],
            cam=result["cam"],
            output_path=args.save_cam,
            alpha=args.cam_alpha,
        )
        print(f"Warstwa Grad-CAM: {result['target_layer']}")
        print(f"Klasa wyjasniana: {result['selected_label']}")
        print(f"Pewnosc: {result['confidence'] * 100:.2f}%")
        if args.threshold is not None:
            print(f"Prog decyzyjny: {args.threshold:.3f}")
        if result["selected_index"] != result["predicted_index"]:
            print(f"Klasa przewidziana przez model: {result['predicted_label']}")
        print(f"Zapisano Grad-CAM: {args.save_cam}")
        return

    result = predict_image(bundle=bundle, image=image)
    print(f"Klasa: {result['predicted_label']}")
    print(f"Pewnosc: {result['confidence'] * 100:.2f}%")
    if args.threshold is not None:
        print(f"Prog decyzyjny: {args.threshold:.3f}")


if __name__ == "__main__":
    main()
