from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

SPLIT_NAMES = {"train", "val", "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatyczny podzial datasetu na train/val/test"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Folder z danymi. Moze to byc surowy katalog klas "
            "albo gotowy uklad train/test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Folder wyjsciowy ze struktura train/val/test. "
            "Dla ukladu train/test mozna pominac, aby przygotowac dataset w miejscu."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "raw", "pre_split"],
        default="auto",
        help="Typ danych wejsciowych: auto, surowe klasy albo gotowy train/test.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Udzial treningu.")
    parser.add_argument("--val", type=float, default=0.15, help="Udzial walidacji.")
    parser.add_argument("--test", type=float, default=0.15, help="Udzial testu.")
    parser.add_argument(
        "--val-from-train",
        type=float,
        default=0.15,
        help=(
            "Dla ukladu train/test: jaki procent plikow z train wydzielic do val."
        ),
    )
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


def detect_layout(input_dir: Path) -> str:
    child_dir_names = {path.name for path in input_dir.iterdir() if path.is_dir()}
    if {"train", "test"}.issubset(child_dir_names):
        return "pre_split"
    return "raw"


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def choose_val_count(total: int, val_ratio: float) -> int:
    if total < 2 or val_ratio <= 0:
        return 0

    val_count = int(total * val_ratio)
    if val_count == 0:
        val_count = 1
    return min(val_count, total - 1)


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


def get_class_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and path.name not in SPLIT_NAMES],
        key=lambda path: path.name,
    )


def get_split_class_dirs(root_dir: Path, split_name: str) -> list[Path]:
    split_dir = root_dir / split_name
    if not split_dir.exists():
        return []
    return sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda path: path.name)


def same_path(path_a: Path, path_b: Path) -> bool:
    return path_a.resolve() == path_b.resolve()


def prepare_raw_layout(args, extensions: set[str]):
    output_dir = args.output_dir or Path("data/real_vs_ai")
    ratios_sum = args.train + args.val + args.test
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("Suma train/val/test musi wynosic 1.0")

    class_dirs = get_class_dirs(args.input_dir)
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
        train_count, val_count, _ = split_counts(len(files), args.train, args.val)

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


def prepare_pre_split_layout(args, extensions: set[str]):
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    in_place = same_path(input_dir, output_dir)

    if in_place and args.copy:
        raise ValueError(
            "Dla pracy w miejscu --copy jest zablokowane, bo utworzyloby duplikaty miedzy train i val."
        )

    if not 0 <= args.val_from_train < 1:
        raise ValueError("--val-from-train musi byc z zakresu [0, 1).")

    train_class_dirs = get_split_class_dirs(input_dir, "train")
    test_class_dirs = get_split_class_dirs(input_dir, "test")
    if not train_class_dirs or not test_class_dirs:
        raise ValueError("Dla ukladu train/test potrzebne sa oba foldery: train i test.")

    train_class_names = {path.name for path in train_class_dirs}
    test_class_names = {path.name for path in test_class_dirs}
    if train_class_names != test_class_names:
        raise ValueError(
            "Foldery train i test musza zawierac te same klasy. "
            f"train={sorted(train_class_names)}, test={sorted(test_class_names)}"
        )

    existing_val_dirs = get_split_class_dirs(input_dir, "val")
    if in_place and existing_val_dirs:
        raise ValueError(
            "Wejsciowy dataset ma juz folder val. "
            "Jesli chcesz go zachowac, uzyj go bez dalszego dzielenia."
        )

    class_names = sorted(train_class_names)
    if in_place:
        ensure_output_dirs(output_dir, ["val"], class_names)
    else:
        ensure_output_dirs(output_dir, ["train", "val", "test"], class_names)

    random.seed(args.seed)

    for class_name in class_names:
        train_class_dir = input_dir / "train" / class_name
        test_class_dir = input_dir / "test" / class_name
        train_files = collect_files(train_class_dir, extensions)
        test_files = collect_files(test_class_dir, extensions)

        if not train_files:
            print(f"[WARN] Klasa '{class_name}' bez plikow w train.")
            continue

        random.shuffle(train_files)
        val_count = choose_val_count(len(train_files), args.val_from_train)
        val_files = train_files[:val_count]
        kept_train_files = train_files[val_count:]

        if in_place:
            for source in val_files:
                relative = source.relative_to(train_class_dir)
                destination = output_dir / "val" / class_name / relative
                copy_or_move(source, destination, copy=False)
        else:
            for source in kept_train_files:
                relative = source.relative_to(train_class_dir)
                destination = output_dir / "train" / class_name / relative
                copy_or_move(source, destination, copy=args.copy)

            for source in val_files:
                relative = source.relative_to(train_class_dir)
                destination = output_dir / "val" / class_name / relative
                copy_or_move(source, destination, copy=args.copy)

            for source in test_files:
                relative = source.relative_to(test_class_dir)
                destination = output_dir / "test" / class_name / relative
                copy_or_move(source, destination, copy=args.copy)

        print(
            f"Klasa '{class_name}': "
            f"train={len(kept_train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    print(f"Zakonczono przygotowanie danych. Wyjscie: {output_dir}")


def main():
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu: {input_dir}")

    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    layout = detect_layout(input_dir) if args.layout == "auto" else args.layout

    if layout == "raw":
        prepare_raw_layout(args, extensions)
        return

    if layout == "pre_split":
        prepare_pre_split_layout(args, extensions)
        return

    raise ValueError(f"Nieobslugiwany layout: {layout}")


if __name__ == "__main__":
    main()
