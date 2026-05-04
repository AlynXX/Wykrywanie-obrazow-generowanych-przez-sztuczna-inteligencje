from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import create_image_folder_dataset
from .inference import choose_positive_class_index, load_model_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Znajdz najlepszy prog decyzyjny dla binarnego modelu fake/real."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--tune-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--metric", type=str, default="f1_positive", choices=["accuracy", "f1_macro", "f1_positive", "balanced_accuracy", "recall_positive"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Opcjonalny JSON z raportem strojenia progu.",
    )
    return parser.parse_args()


def value_or_default(cli_value, config_section: dict, config_key: str, fallback_value):
    if cli_value is not None:
        return cli_value
    return config_section.get(config_key, fallback_value)


def build_loader(dataset, batch_size: int, num_workers: int):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def collect_split_scores(bundle: dict, split_root: Path, batch_size: int, num_workers: int):
    dataset = create_image_folder_dataset(
        root=split_root,
        image_size=bundle["image_size"],
        train=False,
    )
    if dataset.classes != bundle["class_names"]:
        raise ValueError(
            "Klasy w checkpointcie nie zgadzaja sie z datasetem. "
            f"checkpoint={bundle['class_names']}, dataset={dataset.classes}"
        )

    positive_index = choose_positive_class_index(dataset.classes)
    if positive_index is None:
        raise ValueError("Strojenie progu wymaga problemu binarnego fake/real.")

    loader = build_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model = bundle["model"]
    device = bundle["device"]
    model.eval()

    all_labels = []
    all_positive_scores = []
    for images, labels in tqdm(loader, desc=split_root.name, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        all_labels.append(labels.cpu())
        all_positive_scores.append(probabilities[:, positive_index].detach().cpu())

    labels_tensor = torch.cat(all_labels).to(dtype=torch.int64)
    scores_tensor = torch.cat(all_positive_scores).to(dtype=torch.float32)
    return dataset.classes, positive_index, labels_tensor, scores_tensor


def compute_metrics_for_threshold(
    labels: torch.Tensor,
    positive_scores: torch.Tensor,
    positive_index: int,
    threshold: float,
):
    predicted_positive = positive_scores >= threshold
    negative_index = 1 - positive_index
    predictions = torch.where(
        predicted_positive,
        torch.full_like(labels, positive_index),
        torch.full_like(labels, negative_index),
    )

    positive_labels = labels == positive_index
    negative_labels = ~positive_labels
    tp = int((predicted_positive & positive_labels).sum().item())
    fp = int((predicted_positive & negative_labels).sum().item())
    fn = int((~predicted_positive & positive_labels).sum().item())
    tn = int((~predicted_positive & negative_labels).sum().item())

    accuracy = float((predictions == labels).float().mean().item())
    precision_positive = tp / (tp + fp) if (tp + fp) else 0.0
    recall_positive = tp / (tp + fn) if (tp + fn) else 0.0
    f1_positive = (
        2 * precision_positive * recall_positive / (precision_positive + recall_positive)
        if (precision_positive + recall_positive)
        else 0.0
    )

    precision_negative = tn / (tn + fn) if (tn + fn) else 0.0
    recall_negative = tn / (tn + fp) if (tn + fp) else 0.0
    f1_negative = (
        2 * precision_negative * recall_negative / (precision_negative + recall_negative)
        if (precision_negative + recall_negative)
        else 0.0
    )
    f1_macro = (f1_positive + f1_negative) / 2.0
    balanced_accuracy = (recall_positive + recall_negative) / 2.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "f1_positive": f1_positive,
        "precision_negative": precision_negative,
        "recall_negative": recall_negative,
        "f1_negative": f1_negative,
        "f1_macro": f1_macro,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def generate_thresholds(min_value: float, max_value: float, step: float):
    if step <= 0:
        raise ValueError("--threshold-step musi byc dodatni.")
    thresholds = []
    current = min_value
    while current <= max_value + 1e-9:
        thresholds.append(round(current, 6))
        current += step
    return thresholds


def pick_best_threshold(metric_name: str, candidate_metrics: list[dict]):
    ranked = sorted(
        candidate_metrics,
        key=lambda item: (
            item[metric_name],
            item["balanced_accuracy"],
            item["f1_positive"],
            item["accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
        reverse=True,
    )
    return ranked[0], ranked[:10]


def default_output_path(checkpoint_path: Path, metric: str, tune_split: str):
    return Path("reports") / "threshold_tuning" / f"{checkpoint_path.stem}__{tune_split}__{metric}.json"


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    resolved_data_dir = Path(value_or_default(args.data_dir, data_config, "data_dir", "data/real_vs_ai"))
    tune_root = resolved_data_dir / args.tune_split
    eval_root = resolved_data_dir / args.eval_split
    if not tune_root.exists():
        raise FileNotFoundError(f"Nie znaleziono splitu do strojenia progu: {tune_root}")
    if not eval_root.exists():
        raise FileNotFoundError(f"Nie znaleziono splitu do ewaluacji progu: {eval_root}")

    bundle = load_model_bundle(args.checkpoint)
    class_names, positive_index, tune_labels, tune_scores = collect_split_scores(
        bundle=bundle,
        split_root=tune_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, _, eval_labels, eval_scores = collect_split_scores(
        bundle=bundle,
        split_root=eval_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    thresholds = generate_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)
    tune_candidates = [
        compute_metrics_for_threshold(tune_labels, tune_scores, positive_index, threshold)
        for threshold in thresholds
    ]
    best_tune_metrics, top_tune_metrics = pick_best_threshold(args.metric, tune_candidates)
    eval_metrics = compute_metrics_for_threshold(
        eval_labels,
        eval_scores,
        positive_index,
        best_tune_metrics["threshold"],
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "data_dir": str(resolved_data_dir),
        "class_names": class_names,
        "positive_class": class_names[positive_index],
        "metric": args.metric,
        "tune_split": args.tune_split,
        "eval_split": args.eval_split,
        "best_threshold": best_tune_metrics["threshold"],
        "tune_metrics": best_tune_metrics,
        "eval_metrics": eval_metrics,
        "top_thresholds": top_tune_metrics,
    }

    output_path = args.output_path or default_output_path(
        checkpoint_path=args.checkpoint,
        metric=args.metric,
        tune_split=args.tune_split,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print(f"Najlepszy prog dla {args.metric}: {best_tune_metrics['threshold']:.3f}")
    print(
        "Tune split: "
        f"accuracy={best_tune_metrics['accuracy']:.4f}, "
        f"f1_macro={best_tune_metrics['f1_macro']:.4f}, "
        f"f1_positive={best_tune_metrics['f1_positive']:.4f}, "
        f"balanced_accuracy={best_tune_metrics['balanced_accuracy']:.4f}"
    )
    print(
        "Eval split: "
        f"accuracy={eval_metrics['accuracy']:.4f}, "
        f"f1_macro={eval_metrics['f1_macro']:.4f}, "
        f"f1_positive={eval_metrics['f1_positive']:.4f}, "
        f"balanced_accuracy={eval_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Zapisano raport: {output_path}")


if __name__ == "__main__":
    main()
