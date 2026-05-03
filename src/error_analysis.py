from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from .config import load_config
from .dataset import create_image_folder_dataset, robust_pil_loader
from .inference import load_model_bundle, predict_image, predict_with_grad_cam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza bledow i eksport przykladowych predykcji"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Ktory split przeanalizowac.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/error_analysis"),
        help="Folder na raporty i przykladowe eksporty.",
    )
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=12,
        help="Ile przykladow zapisac w kazdej grupie.",
    )
    parser.add_argument(
        "--save-grad-cam",
        action="store_true",
        help="Dla wyeksportowanych przykladow zapisz tez nakladke Grad-CAM.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Opcjonalna warstwa dla Grad-CAM.",
    )
    parser.add_argument(
        "--cam-alpha",
        type=float,
        default=0.45,
        help="Przezroczystosc nakladki Grad-CAM.",
    )
    parser.add_argument(
        "--filter-class",
        type=str,
        default="all",
        help="Filtruj przypadki, gdzie klasa jest rzeczywista lub przewidziana.",
    )
    parser.add_argument(
        "--sort-mode",
        type=str,
        default="default",
        choices=["default", "hardest", "most_confident", "least_confident"],
        help="Sposob sortowania przypadkow przed eksportem.",
    )
    return parser.parse_args()


def value_or_default(cli_value, config_section: dict, config_key: str, fallback_value):
    if cli_value is not None:
        return cli_value
    return config_section.get(config_key, fallback_value)


def choose_positive_class_index(class_names: list[str]) -> int | None:
    if len(class_names) != 2:
        return None

    preferred_keywords = ("fake", "ai", "generated", "synthetic", "deepfake")
    for index, class_name in enumerate(class_names):
        lower_name = class_name.lower()
        if any(keyword in lower_name for keyword in preferred_keywords):
            return index

    return 1


def compute_macro_metrics(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (labels == class_index)).sum().item()
        false_positive = ((predictions == class_index) & (labels != class_index)).sum().item()
        false_negative = ((predictions != class_index) & (labels == class_index)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    accuracy = (predictions == labels).float().mean().item()
    return {
        "accuracy": accuracy,
        "precision_macro": sum(precision_scores) / num_classes,
        "recall_macro": sum(recall_scores) / num_classes,
        "f1_macro": sum(f1_scores) / num_classes,
    }


def compute_binary_roc_auc(binary_labels: torch.Tensor, positive_scores: torch.Tensor) -> float | None:
    positives = int(binary_labels.sum().item())
    negatives = int(binary_labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return None

    ranked_pairs = sorted(
        zip(positive_scores.tolist(), binary_labels.tolist()),
        key=lambda item: item[0],
    )

    rank_sum_positive = 0.0
    rank = 1
    index = 0
    total_items = len(ranked_pairs)

    while index < total_items:
        tie_end = index
        while tie_end + 1 < total_items and ranked_pairs[tie_end + 1][0] == ranked_pairs[index][0]:
            tie_end += 1

        average_rank = (rank + tie_end + 1) / 2.0
        for current_index in range(index, tie_end + 1):
            if ranked_pairs[current_index][1] == 1:
                rank_sum_positive += average_rank

        rank = tie_end + 2
        index = tie_end + 1

    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def build_metrics(records: list[dict], class_names: list[str]):
    labels = torch.tensor([record["actual_index"] for record in records], dtype=torch.int64)
    predictions = torch.tensor([record["predicted_index"] for record in records], dtype=torch.int64)
    probabilities = torch.tensor(
        [record["probability_vector"] for record in records],
        dtype=torch.float32,
    )

    metrics = compute_macro_metrics(labels, predictions, num_classes=len(class_names))
    positive_class_index = choose_positive_class_index(class_names)
    if positive_class_index is not None:
        binary_labels = (labels == positive_class_index).to(dtype=torch.int64)
        roc_auc = compute_binary_roc_auc(binary_labels, probabilities[:, positive_class_index])
        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc
            metrics["positive_class"] = class_names[positive_class_index]
    return metrics


def build_confusion_matrix(records: list[dict], class_names: list[str]):
    matrix = {actual_name: {pred_name: 0 for pred_name in class_names} for actual_name in class_names}
    for record in records:
        matrix[record["actual_label"]][record["predicted_label"]] += 1
    return matrix


def make_safe_name(value: str) -> str:
    safe_characters = []
    for character in value.lower():
        if character.isalnum():
            safe_characters.append(character)
        elif character in {"-", "_"}:
            safe_characters.append(character)
        else:
            safe_characters.append("_")
    return "".join(safe_characters).strip("_") or "sample"


def normalize_filter_class(filter_class: str | None):
    if filter_class is None:
        return None

    normalized = filter_class.strip()
    if not normalized or normalized.lower() == "all":
        return None
    return normalized


def filter_records_by_class(records: list[dict], filter_class: str | None):
    normalized_filter = normalize_filter_class(filter_class)
    if normalized_filter is None:
        return records

    return [
        record
        for record in records
        if record["actual_label"] == normalized_filter or record["predicted_label"] == normalized_filter
    ]


def compute_record_difficulty(record: dict) -> float:
    return record["confidence"] if not record["is_correct"] else 1.0 - record["confidence"]


def sort_group_records(group_name: str, group_records: list[dict], sort_mode: str):
    normalized_mode = sort_mode.strip().lower()

    if normalized_mode == "most_confident":
        return sorted(group_records, key=lambda item: item["confidence"], reverse=True)
    if normalized_mode == "least_confident":
        return sorted(group_records, key=lambda item: item["confidence"])
    if normalized_mode == "hardest":
        return sorted(
            group_records,
            key=lambda item: (compute_record_difficulty(item), item["confidence"]),
            reverse=True,
        )

    if group_name == "correct":
        return sorted(group_records, key=lambda item: item["confidence"])
    return sorted(group_records, key=lambda item: item["confidence"], reverse=True)


def build_record_groups(records: list[dict], class_names: list[str], sort_mode: str):
    positive_class_index = choose_positive_class_index(class_names)

    if positive_class_index is None:
        return {
            "errors": sort_group_records(
                "errors",
                [record for record in records if not record["is_correct"]],
                sort_mode,
            ),
            "correct": sort_group_records(
                "correct",
                [record for record in records if record["is_correct"]],
                sort_mode,
            ),
        }

    groups = {
        "false_positive": [],
        "false_negative": [],
        "true_positive": [],
        "true_negative": [],
    }

    for record in records:
        actual_positive = record["actual_index"] == positive_class_index
        predicted_positive = record["predicted_index"] == positive_class_index

        if predicted_positive and not actual_positive:
            groups["false_positive"].append(record)
        elif actual_positive and not predicted_positive:
            groups["false_negative"].append(record)
        elif actual_positive and predicted_positive:
            groups["true_positive"].append(record)
        else:
            groups["true_negative"].append(record)

    for group_name, group_records in groups.items():
        groups[group_name] = sort_group_records(group_name, group_records, sort_mode)

    return groups


def build_output_tag(filter_class: str | None, sort_mode: str):
    normalized_filter = normalize_filter_class(filter_class)
    if normalized_filter is None and sort_mode == "default":
        return None

    filter_tag = "all" if normalized_filter is None else make_safe_name(normalized_filter)
    return f"{filter_tag}__{sort_mode}"


def export_predictions_csv(output_path: Path, records: list[dict], class_names: list[str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"prob_{class_name}" for class_name in class_names]
    fieldnames = [
        "sample_id",
        "split",
        "source_path",
        "actual_label",
        "predicted_label",
        "confidence",
        "is_correct",
        "width",
        "height",
        "blur_score",
        "contrast_score",
        "quality_warning",
        "quality_severity",
    ] + probability_columns

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "source_path": record["source_path"],
                "actual_label": record["actual_label"],
                "predicted_label": record["predicted_label"],
                "confidence": f"{record['confidence']:.8f}",
                "is_correct": record["is_correct"],
                "width": record["quality"]["width"],
                "height": record["quality"]["height"],
                "blur_score": f"{record['quality']['blur_score']:.4f}",
                "contrast_score": f"{record['quality']['contrast_score']:.4f}",
                "quality_warning": record["quality"]["warning"],
                "quality_severity": record["quality"]["severity"],
            }
            for class_name, probability in zip(class_names, record["probability_vector"]):
                row[f"prob_{class_name}"] = f"{probability:.8f}"
            writer.writerow(row)


def export_group_examples(
    *,
    bundle: dict,
    dataset_root: Path,
    examples_root: Path,
    groups: dict[str, list[dict]],
    examples_per_group: int,
    save_grad_cam: bool,
    target_layer_name: str | None,
    cam_alpha: float,
):
    manifest = {}

    for group_name, group_records in groups.items():
        selected_records = group_records[:examples_per_group]
        manifest[group_name] = []
        group_dir = examples_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for export_index, record in enumerate(selected_records, start=1):
            source_path = Path(record["source_path"])
            sample_slug = make_safe_name(record["sample_id"])
            base_name = (
                f"{export_index:02d}"
                f"__gt_{make_safe_name(record['actual_label'])}"
                f"__pred_{make_safe_name(record['predicted_label'])}"
                f"__{sample_slug}"
            )
            original_output = group_dir / f"{base_name}{source_path.suffix.lower()}"
            shutil.copy2(source_path, original_output)

            export_entry = {
                "sample_id": record["sample_id"],
                "source_path": record["source_path"],
                "relative_path": str(source_path.relative_to(dataset_root)),
                "actual_label": record["actual_label"],
                "predicted_label": record["predicted_label"],
                "confidence": record["confidence"],
                "difficulty_score": record["difficulty_score"],
                "copied_image": str(original_output),
            }

            if save_grad_cam:
                image = robust_pil_loader(str(source_path))
                cam_result = predict_with_grad_cam(
                    bundle=bundle,
                    image=image,
                    target_layer_name=target_layer_name,
                    target_class=record["predicted_index"],
                    cam_alpha=cam_alpha,
                )
                grad_cam_output = group_dir / f"{base_name}__gradcam.jpg"
                cam_result["overlay_image"].save(grad_cam_output)
                export_entry["grad_cam_image"] = str(grad_cam_output)
                export_entry["target_layer"] = cam_result["target_layer"]

            manifest[group_name].append(export_entry)

    return manifest


def analyze_split(bundle: dict, dataset_root: Path, split_name: str):
    dataset = create_image_folder_dataset(
        root=dataset_root,
        image_size=bundle["image_size"],
        train=False,
    )
    records = []

    for sample_index, (source_path, actual_index) in enumerate(dataset.samples):
        image = robust_pil_loader(source_path)
        prediction = predict_image(bundle=bundle, image=image)
        probability_vector = [0.0] * len(dataset.classes)
        for item in prediction["probabilities"]:
            probability_vector[dataset.class_to_idx[item["label"]]] = item["probability"]

        source_path_obj = Path(source_path)
        relative_path = source_path_obj.relative_to(dataset_root)
        records.append(
            {
                "sample_index": sample_index,
                "sample_id": str(relative_path).replace("\\", "/"),
                "split": split_name,
                "source_path": str(source_path_obj),
                "actual_index": int(actual_index),
                "actual_label": dataset.classes[actual_index],
                "predicted_index": int(prediction["predicted_index"]),
                "predicted_label": prediction["predicted_label"],
                "confidence": float(prediction["confidence"]),
                "is_correct": int(actual_index) == int(prediction["predicted_index"]),
                "probability_vector": probability_vector,
                "quality": prediction["quality"],
            }
        )

    return dataset.classes, records


def save_summary(output_path: Path, payload: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2)


def run_error_analysis(
    *,
    checkpoint_path: Path,
    config_path: Path,
    data_dir: Path | None,
    split: str,
    output_dir: Path,
    examples_per_group: int,
    save_grad_cam: bool,
    target_layer_name: str | None,
    cam_alpha: float,
    filter_class: str | None = None,
    sort_mode: str = "default",
):
    config = load_config(config_path)
    data_config = config.get("data", {})

    resolved_data_dir = Path(value_or_default(data_dir, data_config, "data_dir", "data/real_vs_ai"))
    split_root = resolved_data_dir / split
    if not split_root.exists():
        raise FileNotFoundError(f"Nie znaleziono splitu do analizy: {split_root}")

    bundle = load_model_bundle(checkpoint_path)
    class_names, records = analyze_split(bundle=bundle, dataset_root=split_root, split_name=split)
    if not records:
        raise ValueError(f"Split '{split}' nie zawiera zadnych obrazow do analizy.")

    if class_names != bundle["class_names"]:
        raise ValueError(
            "Klasy w checkpointcie nie zgadzaja sie z datasetem. "
            f"checkpoint={bundle['class_names']}, dataset={class_names}"
        )

    normalized_filter = normalize_filter_class(filter_class)
    if normalized_filter is not None and normalized_filter not in class_names:
        raise ValueError(
            f"Nieznana klasa do filtrowania: {normalized_filter}. Dostepne: {', '.join(class_names)}"
        )

    filtered_records = filter_records_by_class(records, normalized_filter)
    if not filtered_records:
        raise ValueError("Brak przypadkow spelniajacych wybrany filtr klasy.")

    tagged_records = []
    for record in filtered_records:
        tagged_record = dict(record)
        tagged_record["difficulty_score"] = compute_record_difficulty(record)
        tagged_records.append(tagged_record)

    output_root = output_dir / split
    output_tag = build_output_tag(normalized_filter, sort_mode)
    examples_root = output_root / "examples"
    if output_tag is not None:
        examples_root = examples_root / output_tag
    metrics = build_metrics(tagged_records, class_names)
    confusion_matrix = build_confusion_matrix(tagged_records, class_names)
    groups = build_record_groups(tagged_records, class_names, sort_mode)
    examples_manifest = export_group_examples(
        bundle=bundle,
        dataset_root=split_root,
        examples_root=examples_root,
        groups=groups,
        examples_per_group=examples_per_group,
        save_grad_cam=save_grad_cam,
        target_layer_name=target_layer_name,
        cam_alpha=cam_alpha,
    )

    predictions_filename = "predictions.csv" if output_tag is None else f"predictions__{output_tag}.csv"
    summary_filename = "summary.json" if output_tag is None else f"summary__{output_tag}.json"
    predictions_csv_path = output_root / predictions_filename
    export_predictions_csv(predictions_csv_path, tagged_records, class_names)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "dataset_root": str(split_root),
        "class_names": class_names,
        "filter_class": normalized_filter or "all",
        "sort_mode": sort_mode,
        "num_samples": len(tagged_records),
        "num_errors": sum(1 for record in tagged_records if not record["is_correct"]),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "groups": {group_name: len(group_records) for group_name, group_records in groups.items()},
        "examples_manifest": examples_manifest,
        "predictions_csv": str(predictions_csv_path),
    }
    summary_path = output_root / summary_filename
    save_summary(summary_path, summary)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "predictions_csv_path": predictions_csv_path,
        "output_root": output_root,
    }


def main():
    args = parse_args()
    result = run_error_analysis(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        examples_per_group=args.examples_per_group,
        save_grad_cam=args.save_grad_cam,
        target_layer_name=args.target_layer,
        cam_alpha=args.cam_alpha,
        filter_class=args.filter_class,
        sort_mode=args.sort_mode,
    )
    summary = result["summary"]
    print(f"Zapisano raport predykcji: {result['predictions_csv_path']}")
    print(f"Zapisano podsumowanie analizy: {result['summary_path']}")
    print(f"Liczba probek: {summary['num_samples']} | bledow: {summary['num_errors']}")
    metric_fragments = [f"{key}={value:.4f}" for key, value in summary["metrics"].items() if isinstance(value, float)]
    if metric_fragments:
        print("Metryki: " + ", ".join(metric_fragments))


if __name__ == "__main__":
    main()
