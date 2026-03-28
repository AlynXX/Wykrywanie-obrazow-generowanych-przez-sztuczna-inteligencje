from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .auto_batch import resolve_amp_dtype
from .config import load_config
from .dataset import create_image_folder_dataset, list_quality_profiles
from .inference import load_model_bundle
from .train import build_loader_kwargs, evaluate, format_metrics

DEFAULT_CHECKPOINT = Path("models/best_model.pt")
DEFAULT_OUTPUT = Path("reports/robustness_eval.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Test odpornosci modelu na pogorszenie jakosci obrazu")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Profile jakosci do sprawdzenia, np. clean jpeg_low mixed_quality",
    )
    return parser.parse_args()


def resolve_batch_size(args_batch_size: int | None, config: dict, checkpoint: dict) -> int:
    if args_batch_size is not None:
        return int(args_batch_size)

    checkpoint_batch = checkpoint.get("batch_size")
    if checkpoint_batch is not None:
        return int(checkpoint_batch)

    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 64)
    if isinstance(batch_size, int):
        return batch_size
    return 64


def resolve_num_workers(args_num_workers: int | None, config: dict) -> int:
    if args_num_workers is not None:
        return int(args_num_workers)
    if os.name == "nt":
        return 0
    return int(config.get("training", {}).get("num_workers", 0))


def resolve_prefetch_factor(config: dict, num_workers: int) -> int | None:
    if num_workers <= 0:
        return None
    value = config.get("training", {}).get("prefetch_factor", 2)
    if value is None:
        return None
    return max(1, int(value))


def ensure_profiles(requested_profiles: list[str] | None):
    available_profiles = list_quality_profiles()
    if not requested_profiles:
        return available_profiles

    normalized = [profile.strip().lower() for profile in requested_profiles]
    unknown = [profile for profile in normalized if profile not in available_profiles]
    if unknown:
        raise ValueError(
            f"Nieznane profile: {', '.join(unknown)}. Dostepne: {', '.join(available_profiles)}"
        )
    return normalized


def maybe_subset_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def create_loader(dataset, batch_size: int, num_workers: int, persistent_workers: bool, prefetch_factor: int | None):
    loader_kwargs = build_loader_kwargs(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.get("data", {}).get("data_dir", "data/real_vs_ai"))
    runtime_config = config.get("runtime", {})
    profiles = ensure_profiles(args.profiles)

    bundle = load_model_bundle(args.checkpoint)
    batch_size = resolve_batch_size(args.batch_size, config, bundle["checkpoint"])
    num_workers = resolve_num_workers(args.num_workers, config)
    persistent_workers = bool(config.get("training", {}).get("persistent_workers", num_workers > 0))
    prefetch_factor = resolve_prefetch_factor(config, num_workers)
    device = bundle["device"]
    model = bundle["model"]
    class_names = bundle["class_names"]
    image_size = int(bundle["image_size"])
    split_dir = data_dir / args.split
    amp_enabled = bool(runtime_config.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(runtime_config.get("amp_dtype", "float16"))
    channels_last_enabled = bool(runtime_config.get("channels_last", True)) and device.type == "cuda"

    if not split_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu splitu: {split_dir}")

    criterion = nn.CrossEntropyLoss()
    results = []
    clean_result = None

    for profile in profiles:
        quality_profile = None if profile == "clean" else profile
        dataset = create_image_folder_dataset(
            root=split_dir,
            image_size=image_size,
            train=False,
            quality_profile=quality_profile,
        )
        dataset = maybe_subset_dataset(dataset, args.max_samples)
        try:
            loader = create_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            loss, metrics = evaluate(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                split_name=profile,
                class_names=class_names,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                channels_last_enabled=channels_last_enabled,
            )
        except PermissionError:
            print(
                f"[{profile}] Multiprocessing DataLoader niedostepny, ponawiam z num_workers=0."
            )
            loader = create_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=0,
                persistent_workers=False,
                prefetch_factor=None,
            )
            loss, metrics = evaluate(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                split_name=profile,
                class_names=class_names,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                channels_last_enabled=channels_last_enabled,
            )

        result = {
            "profile": profile,
            "samples": len(dataset),
            "loss": loss,
            **metrics,
        }
        results.append(result)

        if profile == "clean":
            clean_result = result

        print(f"[{profile}] loss={loss:.4f}, {format_metrics(metrics)}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if clean_result is not None:
        baseline_accuracy = float(clean_result["accuracy"])
        baseline_roc_auc = clean_result.get("roc_auc")
        for result in results:
            result["accuracy_drop_vs_clean"] = baseline_accuracy - float(result["accuracy"])
            if baseline_roc_auc is not None and "roc_auc" in result:
                result["roc_auc_drop_vs_clean"] = float(baseline_roc_auc) - float(result["roc_auc"])

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "data_dir": str(data_dir),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "device": str(device),
        "image_size": image_size,
        "channels_last": channels_last_enabled,
        "profiles": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)

    print(f"Zapisano raport odpornosci: {args.output}")


if __name__ == "__main__":
    main()
