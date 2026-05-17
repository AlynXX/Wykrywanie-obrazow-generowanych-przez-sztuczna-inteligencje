from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Przygotuj zewnetrzny benchmark Gemini z fake i dobranych real."
    )
    parser.add_argument(
        "--fake-dir",
        type=Path,
        required=True,
        help="Flat folder z obrazami wygenerowanymi przez Gemini.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Folder z prawdziwymi obrazami do klasy real.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemini_benchmark"),
        help="Folder wyjsciowy benchmarku.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Nazwa splitu benchmarkowego. Domyslnie test.",
    )
    parser.add_argument(
        "--fake-count",
        type=int,
        default=0,
        help="Ile fake uzyc. 0 oznacza wszystkie dostepne obrazy.",
    )
    parser.add_argument(
        "--real-count",
        type=int,
        default=0,
        help="Ile real uzyc. 0 oznacza tyle samo, co fake albo maksimum dostepnych.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Kopiuj pliki zamiast przenosic.")
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


def copy_or_move(source: Path, destination: Path, copy: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def export_class_files(
    *,
    files: list[Path],
    output_dir: Path,
    split_name: str,
    class_name: str,
    copy: bool,
):
    manifest_entries = []
    class_root = output_dir / split_name / class_name
    class_root.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(files, start=1):
        destination = class_root / f"{index:04d}__{source.stem}{source.suffix.lower()}"
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

    random.seed(args.seed)
    random.shuffle(fake_files)
    random.shuffle(real_files)

    selected_fake_count = len(fake_files) if args.fake_count == 0 else args.fake_count
    if selected_fake_count > len(fake_files):
        raise ValueError(
            f"Za malo fake do benchmarku: wymagane={selected_fake_count}, dostepne={len(fake_files)}"
        )
    selected_fake = fake_files[:selected_fake_count]

    selected_real_count = args.real_count
    if selected_real_count == 0:
        selected_real_count = min(len(real_files), len(selected_fake))
    if selected_real_count > len(real_files):
        raise ValueError(
            f"Za malo real do benchmarku: wymagane={selected_real_count}, dostepne={len(real_files)}"
        )
    selected_real = real_files[:selected_real_count]

    manifest_entries = []
    manifest_entries.extend(
        export_class_files(
            files=selected_fake,
            output_dir=args.output_dir,
            split_name=args.split_name,
            class_name="fake",
            copy=args.copy,
        )
    )
    manifest_entries.extend(
        export_class_files(
            files=selected_real,
            output_dir=args.output_dir,
            split_name=args.split_name,
            class_name="real",
            copy=args.copy,
        )
    )

    summary = {
        "fake_dir": str(args.fake_dir),
        "real_dir": str(args.real_dir),
        "output_dir": str(args.output_dir),
        "split_name": args.split_name,
        "copy_mode": bool(args.copy),
        "counts": {
            "fake": len(selected_fake),
            "real": len(selected_real),
        },
        "entries": manifest_entries,
    }
    summary_path = args.output_dir / "gemini_benchmark_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Zapisano benchmark Gemini: {args.output_dir}")
    print(f"Manifest: {summary_path}")
    print(f"{args.split_name}: fake={len(selected_fake)}, real={len(selected_real)}")


if __name__ == "__main__":
    main()
