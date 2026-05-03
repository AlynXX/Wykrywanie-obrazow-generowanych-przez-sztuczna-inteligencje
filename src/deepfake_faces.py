from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

from .face_utils import (
    build_face_detector,
    detect_faces,
    extract_face_crops,
    iter_image_paths,
    load_image_bgr,
    render_face_boxes,
)
from .inference import load_model_bundle, predict_image

DEFAULT_INPUT_DIR = Path("data/deepfake_faces")
DEFAULT_OUTPUT_DIR = Path("data/deepfake_faces_crops")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Narzedzia do przygotowania i analizy datasetu twarzy."
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Wykryj twarze na pojedynczym obrazie i sklasifikuj cropy wybranym checkpointem.",
    )
    inspect_parser.add_argument("--checkpoint", type=Path, required=True)
    inspect_parser.add_argument("--image", type=Path, required=True)
    add_detector_args(inspect_parser)
    inspect_parser.add_argument(
        "--save-crops-dir",
        type=Path,
        default=None,
        help="Opcjonalny folder do zapisania wycietych twarzy.",
    )
    inspect_parser.add_argument(
        "--save-annotated-image",
        type=Path,
        default=None,
        help="Opcjonalny zapis obrazu z zaznaczonym bboxem twarzy i cropu.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Zbuduj dataset cropow twarzy zachowujac strukture split/class.",
    )
    prepare_parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_detector_args(prepare_parser)
    prepare_parser.add_argument(
        "--single-split-name",
        type=str,
        default="train",
        help="Nazwij tak split wyjsciowy, jesli input-dir nie ma katalogow train/val/test.",
    )
    prepare_parser.add_argument(
        "--save-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Format zapisu cropow twarzy.",
    )

    return parser


def add_detector_args(parser: argparse.ArgumentParser):
    parser.add_argument("--cascade-path", type=Path, default=None)
    parser.add_argument("--min-face-size", type=int, default=64)
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument(
        "--selection",
        type=str,
        default="largest",
        choices=["largest", "all"],
        help="Czy zachowac tylko najwieksza twarz, czy wszystkie wykryte twarze.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help="Maksymalna liczba cropow twarzy na jeden obraz. 0 oznacza bez limitu.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.22,
        help="Dodatkowy margines cropu wokol twarzy jako procent szerokosci i wysokosci bboxu.",
    )
    parser.add_argument(
        "--no-square-crop",
        action="store_true",
        help="Nie wymuszaj kwadratowego cropu twarzy.",
    )


def parse_args():
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        parser.exit(1)
    if argv and argv[0] not in {"inspect", "prepare-dataset", "-h", "--help"}:
        argv = ["inspect", *argv]
    return parser.parse_args(argv)


def resolve_dataset_splits(input_dir: Path, single_split_name: str):
    standard_splits = [split_name for split_name in ["train", "val", "test"] if (input_dir / split_name).exists()]
    if standard_splits:
        return [(split_name, input_dir / split_name) for split_name in standard_splits]
    return [(single_split_name, input_dir)]


def collect_class_roots(split_root: Path):
    class_roots = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
    if not class_roots:
        raise FileNotFoundError(f"Nie znaleziono katalogow klas w: {split_root}")
    return class_roots


def detect_and_crop_faces(image_path: Path, detector, args):
    image_bgr = load_image_bgr(image_path)
    detections = detect_faces(
        image_bgr,
        detector=detector,
        min_face_size=args.min_face_size,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
    )
    max_faces = None if args.max_faces == 0 else args.max_faces
    face_records = extract_face_crops(
        image_bgr,
        detections,
        margin_ratio=args.margin_ratio,
        square_crop=not args.no_square_crop,
        selection=args.selection,
        max_faces=max_faces,
    )
    return image_bgr, face_records


def inspect_single_image(args):
    detector = build_face_detector(args.cascade_path)
    bundle = load_model_bundle(args.checkpoint)
    image_bgr, face_records = detect_and_crop_faces(args.image, detector, args)

    if not face_records:
        print("Nie wykryto twarzy.")
        return

    print(f"Wykryto twarze: {len(face_records)}")
    for record in face_records:
        face_image = Image.fromarray(record["image_rgb"])
        prediction = predict_image(bundle=bundle, image=face_image)
        x1, y1, x2, y2 = record["crop_bbox_xyxy"]
        print(
            f"Twarz {record['face_index']}: crop=({x1}, {y1}, {x2}, {y2}), "
            f"klasa={prediction['predicted_label']}, pewnosc={prediction['confidence'] * 100:.2f}%"
        )

        if args.save_crops_dir is not None:
            args.save_crops_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.save_crops_dir / f"{args.image.stem}__face_{record['face_index']:02d}.jpg"
            face_image.save(output_path, quality=95)

    if args.save_annotated_image is not None:
        args.save_annotated_image.parent.mkdir(parents=True, exist_ok=True)
        annotated_bgr = render_face_boxes(image_bgr, face_records)
        cv2.imwrite(str(args.save_annotated_image), annotated_bgr)
        print(f"Zapisano podglad bbox: {args.save_annotated_image}")


def save_face_crop(face_record: dict, output_path: Path, save_format: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_image = Image.fromarray(face_record["image_rgb"])
    if save_format == "jpg":
        face_image.save(output_path, quality=95)
    else:
        face_image.save(output_path)


def prepare_face_dataset(args):
    detector = build_face_detector(args.cascade_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    aggregate_stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "selection": args.selection,
        "max_faces": args.max_faces,
        "margin_ratio": args.margin_ratio,
        "square_crop": not args.no_square_crop,
        "splits": {},
        "total_images": 0,
        "images_with_faces": 0,
        "images_without_faces": 0,
        "saved_face_crops": 0,
    }

    split_specs = resolve_dataset_splits(args.input_dir, args.single_split_name)
    for split_name, split_root in split_specs:
        split_stats = {
            "images": 0,
            "images_with_faces": 0,
            "images_without_faces": 0,
            "saved_face_crops": 0,
            "classes": {},
        }
        class_roots = collect_class_roots(split_root)

        for class_root in class_roots:
            class_name = class_root.name
            class_stats = {
                "images": 0,
                "images_with_faces": 0,
                "images_without_faces": 0,
                "saved_face_crops": 0,
            }

            for image_path in iter_image_paths(class_root):
                split_stats["images"] += 1
                class_stats["images"] += 1
                aggregate_stats["total_images"] += 1

                try:
                    _, face_records = detect_and_crop_faces(image_path, detector, args)
                except Exception as error:
                    manifest_entries.append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "source_path": str(image_path),
                            "status": "error",
                            "error": str(error),
                        }
                    )
                    continue

                if not face_records:
                    split_stats["images_without_faces"] += 1
                    class_stats["images_without_faces"] += 1
                    aggregate_stats["images_without_faces"] += 1
                    manifest_entries.append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "source_path": str(image_path),
                            "status": "no_face",
                        }
                    )
                    continue

                split_stats["images_with_faces"] += 1
                class_stats["images_with_faces"] += 1
                aggregate_stats["images_with_faces"] += 1

                relative_path = image_path.relative_to(class_root)
                target_parent = output_dir / split_name / class_name / relative_path.parent
                for face_record in face_records:
                    suffix = ".jpg" if args.save_format == "jpg" else ".png"
                    target_path = target_parent / (
                        f"{image_path.stem}__face_{face_record['face_index']:02d}{suffix}"
                    )
                    save_face_crop(face_record, target_path, args.save_format)

                    split_stats["saved_face_crops"] += 1
                    class_stats["saved_face_crops"] += 1
                    aggregate_stats["saved_face_crops"] += 1
                    manifest_entries.append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "source_path": str(image_path),
                            "output_path": str(target_path),
                            "status": "saved",
                            "face_index": face_record["face_index"],
                            "detected_bbox_xywh": face_record["detected_bbox_xywh"],
                            "crop_bbox_xyxy": face_record["crop_bbox_xyxy"],
                            "crop_width": face_record["width"],
                            "crop_height": face_record["height"],
                        }
                    )

            split_stats["classes"][class_name] = class_stats

        aggregate_stats["splits"][split_name] = split_stats

    manifest_path = output_dir / "face_dataset_manifest.json"
    payload = {
        "summary": aggregate_stats,
        "entries": manifest_entries,
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2)

    print(f"Zapisano manifest datasetu twarzy: {manifest_path}")
    print(
        "Podsumowanie: "
        f"obrazy={aggregate_stats['total_images']}, "
        f"z_twarzami={aggregate_stats['images_with_faces']}, "
        f"bez_twarzy={aggregate_stats['images_without_faces']}, "
        f"cropy={aggregate_stats['saved_face_crops']}"
    )


def main():
    args = parse_args()
    if args.command == "prepare-dataset":
        prepare_face_dataset(args)
        return
    inspect_single_image(args)


if __name__ == "__main__":
    main()
