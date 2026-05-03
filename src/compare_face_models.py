from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .error_analysis import build_confusion_matrix, build_metrics
from .face_utils import build_face_detector, detect_faces, extract_face_crops, iter_image_paths, load_image_bgr
from .inference import load_model_bundle, predict_image
from .dataset import robust_pil_loader

DEFAULT_DATA_DIR = Path("data/deepfake_faces")
DEFAULT_OUTPUT_PATH = Path("reports/face_model_comparison.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Porownaj model globalny i model twarzowy na zbiorze portretow."
    )
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--face-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--cascade-path", type=Path, default=None)
    parser.add_argument("--min-face-size", type=int, default=64)
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--no-square-crop", action="store_true")
    return parser.parse_args()


def validate_class_names(global_bundle: dict, face_bundle: dict):
    global_classes = sorted(global_bundle["class_names"])
    face_classes = sorted(face_bundle["class_names"])
    if global_classes != face_classes:
        raise ValueError(
            "Checkpointy uzywaja roznych klas. "
            f"global={global_bundle['class_names']}, face={face_bundle['class_names']}"
        )
    return global_classes


def build_probability_vector(prediction: dict, class_names: list[str]):
    probability_by_label = {
        item["label"]: float(item["probability"]) for item in prediction["probabilities"]
    }
    return [probability_by_label[class_name] for class_name in class_names]


def build_eval_record(
    *,
    sample_id: str,
    source_path: str,
    split: str,
    actual_label: str,
    actual_index: int,
    prediction: dict,
    class_names: list[str],
):
    predicted_label = prediction["predicted_label"]
    predicted_index = class_names.index(predicted_label)
    return {
        "sample_id": sample_id,
        "source_path": source_path,
        "split": split,
        "actual_label": actual_label,
        "actual_index": actual_index,
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "confidence": float(prediction["confidence"]),
        "is_correct": actual_index == predicted_index,
        "probability_vector": build_probability_vector(prediction, class_names),
    }


def export_csv(csv_path: Path, rows: list[dict], class_names: list[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_path",
        "actual_label",
        "face_detected",
        "global_predicted_label",
        "global_confidence",
        "face_predicted_label",
        "face_confidence",
        "crop_bbox_xyxy",
    ]
    fieldnames.extend([f"global_prob_{class_name}" for class_name in class_names])
    fieldnames.extend([f"face_prob_{class_name}" for class_name in class_names])

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    split_root = args.data_dir / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Nie znaleziono splitu do porownania: {split_root}")

    global_bundle = load_model_bundle(args.global_checkpoint)
    face_bundle = load_model_bundle(args.face_checkpoint)
    class_names = validate_class_names(global_bundle, face_bundle)

    detector = build_face_detector(args.cascade_path)
    class_roots = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
    if not class_roots:
        raise FileNotFoundError(f"Brak katalogow klas w: {split_root}")

    global_all_records = []
    global_detected_records = []
    face_detected_records = []
    csv_rows = []
    total_images = 0
    detected_faces = 0
    missing_faces = 0

    for class_index, class_root in enumerate(class_roots):
        actual_label = class_root.name
        if actual_label not in class_names:
            raise ValueError(
                f"Klasa '{actual_label}' ze zbioru nie wystepuje w checkpointach: {class_names}"
            )
        actual_index = class_names.index(actual_label)

        for image_path in iter_image_paths(class_root):
            total_images += 1
            image = robust_pil_loader(str(image_path))
            global_prediction = predict_image(bundle=global_bundle, image=image)
            relative_path = image_path.relative_to(split_root)
            sample_id = str(relative_path).replace("\\", "/")
            global_record = build_eval_record(
                sample_id=sample_id,
                source_path=str(image_path),
                split=args.split,
                actual_label=actual_label,
                actual_index=actual_index,
                prediction=global_prediction,
                class_names=class_names,
            )
            global_all_records.append(global_record)

            image_bgr = load_image_bgr(image_path)
            detections = detect_faces(
                image_bgr,
                detector=detector,
                min_face_size=args.min_face_size,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
            )
            face_crops = extract_face_crops(
                image_bgr,
                detections,
                margin_ratio=args.margin_ratio,
                square_crop=not args.no_square_crop,
                selection="largest",
                max_faces=1,
            )

            csv_row = {
                "sample_id": sample_id,
                "source_path": str(image_path),
                "actual_label": actual_label,
                "face_detected": bool(face_crops),
                "global_predicted_label": global_prediction["predicted_label"],
                "global_confidence": f"{global_prediction['confidence']:.8f}",
                "face_predicted_label": "",
                "face_confidence": "",
                "crop_bbox_xyxy": "",
            }
            for class_name, probability in zip(
                class_names,
                build_probability_vector(global_prediction, class_names),
            ):
                csv_row[f"global_prob_{class_name}"] = f"{probability:.8f}"
                csv_row[f"face_prob_{class_name}"] = ""

            if not face_crops:
                missing_faces += 1
                csv_rows.append(csv_row)
                continue

            detected_faces += 1
            face_image = robust_pil_loader(str(image_path))
            face_image = face_image.crop(tuple(face_crops[0]["crop_bbox_xyxy"]))
            face_prediction = predict_image(bundle=face_bundle, image=face_image)

            global_detected_records.append(global_record)
            face_detected_records.append(
                build_eval_record(
                    sample_id=sample_id,
                    source_path=str(image_path),
                    split=args.split,
                    actual_label=actual_label,
                    actual_index=actual_index,
                    prediction=face_prediction,
                    class_names=class_names,
                )
            )

            csv_row["face_predicted_label"] = face_prediction["predicted_label"]
            csv_row["face_confidence"] = f"{face_prediction['confidence']:.8f}"
            csv_row["crop_bbox_xyxy"] = ",".join(str(value) for value in face_crops[0]["crop_bbox_xyxy"])
            for class_name, probability in zip(
                class_names,
                build_probability_vector(face_prediction, class_names),
            ):
                csv_row[f"face_prob_{class_name}"] = f"{probability:.8f}"
            csv_rows.append(csv_row)

    if not global_all_records:
        raise ValueError(f"Split '{args.split}' nie zawiera zadnych obrazow.")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv_path or output_path.with_suffix(".csv")
    export_csv(csv_path, csv_rows, class_names)

    disagreements = 0
    for global_record, face_record in zip(global_detected_records, face_detected_records):
        if global_record["predicted_label"] != face_record["predicted_label"]:
            disagreements += 1

    summary = {
        "split": args.split,
        "data_dir": str(args.data_dir),
        "global_checkpoint": str(args.global_checkpoint),
        "face_checkpoint": str(args.face_checkpoint),
        "class_names": class_names,
        "counts": {
            "total_images": total_images,
            "detected_faces": detected_faces,
            "missing_faces": missing_faces,
            "face_detection_rate": detected_faces / total_images,
        },
        "metrics": {
            "global_all": build_metrics(global_all_records, class_names),
            "global_on_detected_subset": build_metrics(global_detected_records, class_names)
            if global_detected_records
            else {},
            "face_on_detected_subset": build_metrics(face_detected_records, class_names)
            if face_detected_records
            else {},
        },
        "confusion_matrices": {
            "global_all": build_confusion_matrix(global_all_records, class_names),
            "global_on_detected_subset": build_confusion_matrix(global_detected_records, class_names)
            if global_detected_records
            else {},
            "face_on_detected_subset": build_confusion_matrix(face_detected_records, class_names)
            if face_detected_records
            else {},
        },
        "comparison": {
            "paired_samples": len(face_detected_records),
            "disagreements": disagreements,
            "agreement_rate": (
                (len(face_detected_records) - disagreements) / len(face_detected_records)
                if face_detected_records
                else 0.0
            ),
        },
        "csv_path": str(csv_path),
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(f"Zapisano raport porownawczy: {output_path}")
    print(f"Zapisano CSV: {csv_path}")
    print(
        "Face detection: "
        f"{detected_faces}/{total_images} ({summary['counts']['face_detection_rate'] * 100:.2f}%)"
    )
    if face_detected_records:
        global_accuracy = summary["metrics"]["global_on_detected_subset"]["accuracy"]
        face_accuracy = summary["metrics"]["face_on_detected_subset"]["accuracy"]
        print(
            "Accuracy na zbiorze z wykryta twarza: "
            f"global={global_accuracy:.4f}, face={face_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
