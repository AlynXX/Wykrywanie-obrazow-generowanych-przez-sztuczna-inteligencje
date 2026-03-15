from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatyczny podzial datasetu na train/val/test"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder z klasami (np. data/raw/ z podfolderami klas).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real_vs_ai"),
        help="Folder wyjsciowy ze strukturą train/val/test.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Udzial treningu.")
    parser.add_argument("--val", type=float, default=0.15, help="Udzial walidacji.")
    parser.add_argument("--test", type=float, default=0.15, help="Udzial testu.")
    parser.add_argument("--seed", type=int, default=42, help="Seed losowania.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Kopiuj pliki zamiast przenosic.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Dozwolone rozszerzenia, rozdzielone przecinkami.",
    )
    return parser.parse_args()


def collect_files(class_dir: Path, extensions: set[str]) -> list[Path]:
    return [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def ensure_output_dirs(output_dir: Path, splits: list[str], class_names: list[str]):
    for split in splits:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_or_move(source: Path, destination: Path, copy: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu: {input_dir}")

    ratios_sum = args.train + args.val + args.test
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("Suma train/val/test musi wynosic 1.0")

    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    class_dirs = [path for path in input_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError("Brak podfolderow klas w input-dir.")

    class_names = [path.name for path in class_dirs]
    ensure_output_dirs(output_dir, ["train", "val", "test"], class_names)

    random.seed(args.seed)

    for class_dir in class_dirs:
        files = collect_files(class_dir, extensions)
        if not files:
            print(f"[WARN] Klasa '{class_dir.name}' bez plikow.")
            continue

        random.shuffle(files)
        train_count, val_count, test_count = split_counts(len(files), args.train, args.val)

        train_files = files[:train_count]
        val_files = files[train_count : train_count + val_count]
        test_files = files[train_count + val_count :]

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            for source in split_files:
                relative = source.relative_to(class_dir)
                destination = output_dir / split_name / class_dir.name / relative
                copy_or_move(source, destination, copy=args.copy)

        print(
            f"Klasa '{class_dir.name}': "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    print(f"Zakonczono podzial danych. Wyjscie: {output_dir}")


if __name__ == "__main__":
    main()
