from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import load_datasets
from .model import create_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


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

    data_dir = Path(value_or_default(args.data_dir, data_config, "data_dir", "data/real_vs_ai"))
    output_dir = Path(value_or_default(args.output_dir, data_config, "output_dir", "models"))
    model_name = value_or_default(args.model_name, model_config, "model_name", "resnet18")
    pretrained = model_config.get("pretrained", True)
    image_size = int(value_or_default(args.image_size, model_config, "image_size", 224))
    epochs = int(value_or_default(args.epochs, training_config, "epochs", 10))
    batch_size = int(value_or_default(args.batch_size, training_config, "batch_size", 32))
    learning_rate = float(
        value_or_default(args.learning_rate, training_config, "learning_rate", 1e-4)
    )
    num_workers = int(value_or_default(args.num_workers, training_config, "num_workers", 2))
    torch_compile_enabled = bool(runtime_config.get("torch_compile", False))
    compile_mode = runtime_config.get("compile_mode", "default")
    compile_backend = runtime_config.get("compile_backend", "inductor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_datasets(data_dir, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    base_model = create_model(
        model_name=model_name,
        num_classes=len(train_dataset.classes),
        pretrained=pretrained,
    ).to(device)
    model = base_model
    if torch_compile_enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(
                base_model,
                mode=compile_mode,
                backend=compile_backend,
            )
            print(
                f"Włączono torch.compile (mode={compile_mode}, backend={compile_backend})."
            )
        except Exception as error:
            print(f"torch.compile niedostępne lub nieudane: {error}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split_name="val",
        )

        print(
            f"Epoka {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": model_name,
                    "class_names": train_dataset.classes,
                    "image_size": image_size,
                    "state_dict": base_model.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Zapisano nowy najlepszy model: {checkpoint_path}")

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(best_checkpoint["state_dict"])

    test_loss, test_acc = evaluate(
        model=base_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="test",
    )

    print(
        f"Koniec treningu. Najlepsze val_acc={best_val_acc:.4f} | "
        f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
