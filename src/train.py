from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .auto_batch import configure_compile_caches, resolve_amp_dtype, resolve_smart_batch_size
from .config import load_config
from .dataset import load_datasets
from .model import create_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    channels_last_enabled: bool,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=non_blocking)
        if channels_last_enabled and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


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


def build_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: list[str],
):
    metrics = compute_macro_metrics(labels, predictions, num_classes=len(class_names))

    positive_class_index = choose_positive_class_index(class_names)
    if positive_class_index is not None:
        binary_labels = (labels == positive_class_index).to(dtype=torch.int64)
        roc_auc = compute_binary_roc_auc(binary_labels, probabilities[:, positive_class_index])
        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc
            metrics["positive_class"] = class_names[positive_class_index]

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    class_names: list[str],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    channels_last_enabled: bool,
):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    non_blocking = device.type == "cuda"

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device, non_blocking=non_blocking)
        if channels_last_enabled and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=non_blocking)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    probabilities_tensor = torch.cat(all_probabilities)
    metrics = build_metrics(labels_tensor, predictions_tensor, probabilities_tensor, class_names)
    return total_loss / total, metrics


def format_metrics(metrics: dict[str, float | str]) -> str:
    ordered_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
    ]
    fragments = []
    for key in ordered_keys:
        if key in metrics:
            fragments.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(fragments)


def save_training_summary(summary_path: Path, payload: dict):
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2)


def build_checkpoint_payload(
    *,
    epoch: int,
    next_epoch: int,
    model_name: str,
    class_names: list[str],
    image_size: int,
    selection_metric: str,
    best_val_score: float,
    batch_size: int,
    state_dict: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    training_history: list[dict],
    early_stopping_state: dict[str, int | float | str | bool] | None = None,
    interrupted: bool = False,
):
    payload = {
        "epoch": epoch,
        "next_epoch": next_epoch,
        "model_name": model_name,
        "class_names": class_names,
        "image_size": image_size,
        "selection_metric": selection_metric,
        "best_val_score": best_val_score,
        "batch_size": batch_size,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "training_history": training_history,
        "interrupted": interrupted,
    }
    if early_stopping_state is not None:
        payload["early_stopping_state"] = early_stopping_state
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def save_checkpoint(checkpoint_path: Path, payload: dict):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_resume_checkpoint(resume_path: Path):
    if not resume_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu do wznowienia: {resume_path}")
    return torch.load(resume_path, map_location="cpu")


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_early_stopping_monitor(monitor_value: str, selection_metric: str) -> str:
    normalized = monitor_value.strip().lower()
    if normalized in {"", "auto"}:
        return selection_metric
    return normalized


@torch.no_grad()
def probe_compiled_model(
    model: nn.Module,
    dataset,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    batch_size: int,
    channels_last_enabled: bool,
):
    loader_kwargs = build_loader_kwargs(
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=None,
    )
    probe_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 2),
        shuffle=False,
        **loader_kwargs,
    )
    images, _ = next(iter(probe_loader))
    images = images.to(device, non_blocking=device.type == "cuda")
    if channels_last_enabled and images.ndim == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        _ = model(images)


def build_loader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
):
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Trening klasyfikatora Real vs AI")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def value_or_default(
    cli_value,
    config_section: dict,
    config_key: str,
    fallback_value,
):
    if cli_value is not None:
        return cli_value
    return config_section.get(config_key, fallback_value)


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    runtime_config = config.get("runtime", {})
    auto_batch_config = config.get("auto_batch", {})
    early_stopping_config = config.get("early_stopping", {})
    resume_checkpoint = load_resume_checkpoint(args.resume) if args.resume is not None else None

    data_dir = Path(value_or_default(args.data_dir, data_config, "data_dir", "data/real_vs_ai"))
    output_dir = Path(value_or_default(args.output_dir, data_config, "output_dir", "models"))
    model_name = value_or_default(args.model_name, model_config, "model_name", "resnet18")
    pretrained = model_config.get("pretrained", True)
    image_size = int(value_or_default(args.image_size, model_config, "image_size", 224))
    epochs = int(value_or_default(args.epochs, training_config, "epochs", 10))
    batch_size_value = value_or_default(args.batch_size, training_config, "batch_size", 32)
    learning_rate = float(
        value_or_default(args.learning_rate, training_config, "learning_rate", 1e-4)
    )
    num_workers = int(value_or_default(args.num_workers, training_config, "num_workers", 2))
    persistent_workers = bool(training_config.get("persistent_workers", num_workers > 0))
    prefetch_factor_value = training_config.get("prefetch_factor", 2)
    prefetch_factor = (
        max(1, int(prefetch_factor_value))
        if prefetch_factor_value is not None and num_workers > 0
        else None
    )
    seed = int(value_or_default(args.seed, training_config, "seed", 42))
    torch_compile_enabled = bool(runtime_config.get("torch_compile", False))
    compile_mode = runtime_config.get("compile_mode", "default")
    compile_backend = runtime_config.get("compile_backend", "inductor")
    amp_enabled = bool(runtime_config.get("amp", True))
    amp_dtype = resolve_amp_dtype(runtime_config.get("amp_dtype", "float16"))
    cudnn_benchmark = bool(runtime_config.get("cudnn_benchmark", True))
    channels_last_enabled = bool(runtime_config.get("channels_last", True))

    if resume_checkpoint is not None:
        model_name = str(resume_checkpoint.get("model_name", model_name))
        image_size = int(resume_checkpoint.get("image_size", image_size))
        print(f"Wznawiam trening z checkpointu: {args.resume}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if torch_compile_enabled:
        configure_compile_caches(output_dir)

    train_dataset, val_dataset, test_dataset = load_datasets(data_dir, image_size)
    class_names = train_dataset.classes
    if resume_checkpoint is not None:
        checkpoint_class_names = list(resume_checkpoint.get("class_names", []))
        if checkpoint_class_names and checkpoint_class_names != class_names:
            raise ValueError(
                "Klasy w checkpointcie nie zgadzaja sie z aktualnym datasetem. "
                f"checkpoint={checkpoint_class_names}, dataset={class_names}"
            )
    auto_batch_enabled = bool(auto_batch_config.get("enabled", False))
    effective_auto_batch_config = dict(auto_batch_config)
    effective_auto_batch_config.setdefault("compile_mode", compile_mode)
    effective_auto_batch_config.setdefault("compile_backend", compile_backend)
    effective_auto_batch_config.setdefault("probe_amp", amp_enabled)

    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    else:
        batch_size_is_auto = (
            isinstance(batch_size_value, str) and batch_size_value.strip().lower() == "auto"
        )
        if resume_checkpoint is not None and "batch_size" in resume_checkpoint:
            batch_size = int(resume_checkpoint["batch_size"])
        elif batch_size_is_auto or auto_batch_enabled:
            batch_size = resolve_smart_batch_size(
                model_name=model_name,
                num_classes=len(class_names),
                image_size=image_size,
                learning_rate=learning_rate,
                device=device,
                compile_value=torch_compile_enabled,
                amp_enabled=amp_enabled and device.type == "cuda",
                amp_dtype=amp_dtype,
                auto_batch_cfg=effective_auto_batch_config,
            )
        else:
            batch_size = int(batch_size_value)

    print(f"Uzywany batch_size={batch_size}")
    if device.type == "cuda":
        print(
            "Loader i runtime: "
            f"num_workers={num_workers}, "
            f"persistent_workers={persistent_workers if num_workers > 0 else False}, "
            f"prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}, "
            f"channels_last={channels_last_enabled}, "
            f"cudnn_benchmark={cudnn_benchmark}"
        )

    loader_kwargs = build_loader_kwargs(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    base_model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=pretrained,
    )
    if channels_last_enabled and device.type == "cuda":
        base_model = base_model.to(memory_format=torch.channels_last)
    base_model = base_model.to(device)

    effective_amp = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=effective_amp and amp_dtype == torch.float16,
    )

    model = base_model
    if torch_compile_enabled and hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(
                base_model,
                mode=compile_mode,
                backend=compile_backend,
            )
            probe_compiled_model(
                model=compiled_model,
                dataset=train_dataset,
                device=device,
                amp_enabled=effective_amp,
                amp_dtype=amp_dtype,
                batch_size=batch_size,
                channels_last_enabled=channels_last_enabled and device.type == "cuda",
            )
            model = compiled_model
            print(
                f"Wlaczono torch.compile (mode={compile_mode}, backend={compile_backend})."
            )
        except Exception as error:
            print(f"torch.compile niedostepne lub nieudane: {error}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)

    selection_metric = "roc_auc" if len(class_names) == 2 else "accuracy"
    early_stopping_enabled = bool(early_stopping_config.get("enabled", True))
    early_stopping_patience = int(early_stopping_config.get("patience", 3))
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.001))
    early_stopping_start_epoch = int(early_stopping_config.get("start_epoch", 3))
    early_stopping_monitor = resolve_early_stopping_monitor(
        str(early_stopping_config.get("monitor", "auto")),
        selection_metric,
    )
    start_epoch = 1
    best_val_score = -1.0
    early_stopping_best_score = float("-inf")
    early_stopping_bad_epochs = 0
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    interrupted_checkpoint_path = output_dir / "interrupted_checkpoint.pt"
    summary_path = output_dir / "training_summary.json"
    training_history = []

    if resume_checkpoint is not None:
        base_model.load_state_dict(resume_checkpoint["state_dict"])
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state is not None and scaler is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)
        selection_metric = str(resume_checkpoint.get("selection_metric", selection_metric))
        best_val_score = float(resume_checkpoint.get("best_val_score", best_val_score))
        start_epoch = int(resume_checkpoint.get("next_epoch", resume_checkpoint.get("epoch", 0) + 1))
        training_history = list(resume_checkpoint.get("training_history", []))
        early_state = resume_checkpoint.get("early_stopping_state", {})
        if isinstance(early_state, dict):
            early_stopping_best_score = float(
                early_state.get("best_score", early_stopping_best_score)
            )
            early_stopping_bad_epochs = int(early_state.get("bad_epochs", early_stopping_bad_epochs))
            best_epoch = int(early_state.get("best_epoch", best_epoch))
        print(
            f"Wznowienie od epoki {start_epoch} | najlepsze val_{selection_metric}={best_val_score:.4f}"
        )
    else:
        early_stopping_best_score = best_val_score

    if early_stopping_monitor == selection_metric:
        early_stopping_best_score = max(early_stopping_best_score, best_val_score)

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=effective_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                channels_last_enabled=channels_last_enabled and device.type == "cuda",
            )
            val_loss, val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                split_name="val",
                class_names=class_names,
                amp_enabled=effective_amp,
                amp_dtype=amp_dtype,
                channels_last_enabled=channels_last_enabled and device.type == "cuda",
            )

            current_val_score = float(val_metrics.get(selection_metric, val_metrics["accuracy"]))
            training_history.append(
                {
                    "epoch": epoch,
                    "train": {"loss": train_loss, "accuracy": train_acc},
                    "val": {"loss": val_loss, **val_metrics},
                }
            )

            print(
                f"Epoka {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, {format_metrics(val_metrics)}"
            )

            monitor_score = float(val_metrics.get(early_stopping_monitor, val_metrics["accuracy"]))
            is_improved = current_val_score > best_val_score
            if is_improved:
                best_val_score = current_val_score

            if epoch >= early_stopping_start_epoch:
                if monitor_score > early_stopping_best_score + early_stopping_min_delta:
                    early_stopping_best_score = monitor_score
                    early_stopping_bad_epochs = 0
                    best_epoch = epoch
                else:
                    early_stopping_bad_epochs += 1
                    print(
                        f"Brak poprawy {early_stopping_monitor} przez {early_stopping_bad_epochs}/"
                        f"{early_stopping_patience} epok."
                    )
            else:
                if monitor_score > early_stopping_best_score:
                    early_stopping_best_score = monitor_score
                    best_epoch = epoch

            early_stopping_state = {
                "enabled": early_stopping_enabled,
                "monitor": early_stopping_monitor,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "start_epoch": early_stopping_start_epoch,
                "best_score": early_stopping_best_score,
                "bad_epochs": early_stopping_bad_epochs,
                "best_epoch": best_epoch,
            }

            last_payload = build_checkpoint_payload(
                epoch=epoch,
                next_epoch=epoch + 1,
                model_name=model_name,
                class_names=class_names,
                image_size=image_size,
                selection_metric=selection_metric,
                best_val_score=best_val_score,
                batch_size=batch_size,
                state_dict=base_model.state_dict(),
                optimizer=optimizer,
                scaler=scaler,
                training_history=training_history,
                early_stopping_state=early_stopping_state,
            )
            save_checkpoint(last_checkpoint_path, last_payload)

            if is_improved:
                best_payload = build_checkpoint_payload(
                    epoch=epoch,
                    next_epoch=epoch + 1,
                    model_name=model_name,
                    class_names=class_names,
                    image_size=image_size,
                    selection_metric=selection_metric,
                    best_val_score=best_val_score,
                    batch_size=batch_size,
                    state_dict=base_model.state_dict(),
                    optimizer=optimizer,
                    scaler=scaler,
                    training_history=training_history,
                    early_stopping_state=early_stopping_state,
                )
                save_checkpoint(checkpoint_path, best_payload)
                print(f"Zapisano nowy najlepszy model: {checkpoint_path}")

            if early_stopping_enabled and epoch >= early_stopping_start_epoch:
                if early_stopping_bad_epochs >= early_stopping_patience:
                    print(
                        f"Early stopping: zatrzymano po epoce {epoch}. "
                        f"Najlepsze {early_stopping_monitor} bylo w epoce {best_epoch}."
                    )
                    break
    except KeyboardInterrupt:
        early_stopping_state = {
            "enabled": early_stopping_enabled,
            "monitor": early_stopping_monitor,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "start_epoch": early_stopping_start_epoch,
            "best_score": early_stopping_best_score,
            "bad_epochs": early_stopping_bad_epochs,
            "best_epoch": best_epoch,
        }
        interrupted_payload = build_checkpoint_payload(
            epoch=max(start_epoch, current_epoch),
            next_epoch=max(start_epoch, current_epoch),
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            selection_metric=selection_metric,
            best_val_score=best_val_score,
            batch_size=batch_size,
            state_dict=base_model.state_dict(),
            optimizer=optimizer,
            scaler=scaler,
            training_history=training_history,
            early_stopping_state=early_stopping_state,
            interrupted=True,
        )
        save_checkpoint(interrupted_checkpoint_path, interrupted_payload)
        print(f"Trening przerwany. Zapisano checkpoint wznowienia: {interrupted_checkpoint_path}")
        return

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(best_checkpoint["state_dict"])

    test_loss, test_metrics = evaluate(
        model=base_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="test",
        class_names=class_names,
        amp_enabled=effective_amp,
        amp_dtype=amp_dtype,
        channels_last_enabled=channels_last_enabled and device.type == "cuda",
    )

    save_training_summary(
        summary_path,
        {
            "model_name": model_name,
            "class_names": class_names,
            "dataset_sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
            "selection_metric": selection_metric,
            "best_val_score": best_val_score,
            "runtime": {
                "device": device.type,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "persistent_workers": persistent_workers if num_workers > 0 else False,
                "prefetch_factor": prefetch_factor if num_workers > 0 else None,
                "seed": seed,
                "torch_compile": torch_compile_enabled,
                "compile_mode": compile_mode,
                "amp": effective_amp,
                "amp_dtype": str(amp_dtype).replace("torch.", ""),
                "cudnn_benchmark": cudnn_benchmark if device.type == "cuda" else False,
                "channels_last": channels_last_enabled if device.type == "cuda" else False,
            },
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "monitor": early_stopping_monitor,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "start_epoch": early_stopping_start_epoch,
                "best_score": early_stopping_best_score,
                "bad_epochs": early_stopping_bad_epochs,
                "best_epoch": best_epoch,
            },
            "history": training_history,
            "test": {"loss": test_loss, **test_metrics},
        },
    )

    print(
        f"Koniec treningu. Najlepsze val_{selection_metric}={best_val_score:.4f} | "
        f"test_loss={test_loss:.4f}, {format_metrics(test_metrics)}"
    )
    print(f"Zapisano podsumowanie treningu: {summary_path}")


if __name__ == "__main__":
    main()
