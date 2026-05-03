from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Przygotuj dataset adaptacyjny z hard fake i starych real."
    )
    parser.add_argument(
        "--fake-dir",
        type=Path,
        required=True,
        help="Folder z trudnymi fake'ami do adaptacji.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Folder z prawdziwymi twarzami ze starego datasetu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hard_fakes_adaptation"),
        help="Folder wyjsciowy ze struktura train/val/test.",
    )
    parser.add_argument("--train-fake", type=int, default=500)
    parser.add_argument("--val-fake", type=int, default=100)
    parser.add_argument("--test-fake", type=int, default=100)
    parser.add_argument("--train-real", type=int, default=1000)
    parser.add_argument("--val-real", type=int, default=200)
    parser.add_argument("--test-real", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Kopiuj pliki zamiast przenosic.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help="Dozwolone rozszerzenia, rozdzielone przecinkami.",
    )
    return parser.parse_args()


def collect_files(root: Path, extensions: set[str]):
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
    )


def ensure_output_dirs(output_dir: Path):
    for split_name in ["train", "val", "test"]:
        for class_name in ["fake", "real"]:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)


def copy_or_move(source: Path, destination: Path, copy: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def select_subset(files: list[Path], count: int, label: str):
    if count < 0:
        raise ValueError(f"Liczba probek dla {label} nie moze byc ujemna.")
    if len(files) < count:
        raise ValueError(
            f"Za malo plikow dla {label}: wymagane={count}, dostepne={len(files)}"
        )
    return files[:count]


def build_splits(
    *,
    fake_files: list[Path],
    real_files: list[Path],
    args,
):
    random.seed(args.seed)
    random.shuffle(fake_files)
    random.shuffle(real_files)

    fake_train = select_subset(fake_files, args.train_fake, "train/fake")
    fake_val = select_subset(fake_files[args.train_fake :], args.val_fake, "val/fake")
    fake_test = select_subset(
        fake_files[args.train_fake + args.val_fake :],
        args.test_fake,
        "test/fake",
    )

    real_train = select_subset(real_files, args.train_real, "train/real")
    real_val = select_subset(real_files[args.train_real :], args.val_real, "val/real")
    real_test = select_subset(
        real_files[args.train_real + args.val_real :],
        args.test_real,
        "test/real",
    )

    return {
        "train": {"fake": fake_train, "real": real_train},
        "val": {"fake": fake_val, "real": real_val},
        "test": {"fake": fake_test, "real": real_test},
    }


def export_split(
    *,
    splits: dict[str, dict[str, list[Path]]],
    output_dir: Path,
    copy: bool,
):
    manifest_entries = []
    for split_name, classes in splits.items():
        for class_name, files in classes.items():
            class_root = output_dir / split_name / class_name
            for index, source in enumerate(files, start=1):
                target_name = f"{index:04d}__{source.stem}{source.suffix.lower()}"
                destination = class_root / target_name
                copy_or_move(source, destination, copy=copy)
                manifest_entries.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "source_path": str(source),
                        "output_path": str(destination),
                    }
                )
    return manifest_entries


def main():
    args = parse_args()
    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}

    if not args.fake_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono fake-dir: {args.fake_dir}")
    if not args.real_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono real-dir: {args.real_dir}")

    fake_files = collect_files(args.fake_dir, extensions)
    real_files = collect_files(args.real_dir, extensions)
    if not fake_files:
        raise ValueError(f"Brak obrazow w fake-dir: {args.fake_dir}")
    if not real_files:
        raise ValueError(f"Brak obrazow w real-dir: {args.real_dir}")

    ensure_output_dirs(args.output_dir)
    splits = build_splits(fake_files=fake_files, real_files=real_files, args=args)
    manifest_entries = export_split(
        splits=splits,
        output_dir=args.output_dir,
        copy=args.copy,
    )

    summary = {
        "fake_dir": str(args.fake_dir),
        "real_dir": str(args.real_dir),
        "output_dir": str(args.output_dir),
        "copy_mode": bool(args.copy),
        "counts": {
            "train": {"fake": len(splits["train"]["fake"]), "real": len(splits["train"]["real"])},
            "val": {"fake": len(splits["val"]["fake"]), "real": len(splits["val"]["real"])},
            "test": {"fake": len(splits["test"]["fake"]), "real": len(splits["test"]["real"])},
        },
        "entries": manifest_entries,
    }

    summary_path = args.output_dir / "adaptation_manifest.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Zapisano dataset adaptacyjny: {args.output_dir}")
    print(f"Manifest: {summary_path}")
    for split_name in ["train", "val", "test"]:
        counts = summary["counts"][split_name]
        print(
            f"{split_name}: fake={counts['fake']}, real={counts['real']}"
        )


if __name__ == "__main__":
    main()
