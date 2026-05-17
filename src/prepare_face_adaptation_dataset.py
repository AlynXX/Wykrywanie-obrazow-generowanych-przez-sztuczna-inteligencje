from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from .config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zloz multisource dataset adaptacyjny do fine-tuningu modelu twarzowego."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("face_adaptation_sources.yaml"),
        help="Plik YAML ze zrodlami fake/real i limitami na zrodlo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Opcjonalne nadpisanie output_dir ze specyfikacji.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Kopiuj pliki zamiast przenosic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Opcjonalne nadpisanie seeda ze specyfikacji.",
    )
    return parser.parse_args()


def resolve_root_path(base_dir: Path, raw_path: str):
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


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


def select_subset(files: list[Path], limit: int | None, *, label: str, rng: random.Random):
    shuffled = list(files)
    rng.shuffle(shuffled)

    if limit is None:
        return shuffled
    if limit <= 0:
        raise ValueError(f"Limit dla {label} musi byc dodatni albo pusty.")
    if len(shuffled) < limit:
        raise ValueError(
            f"Za malo plikow dla {label}: wymagane={limit}, dostepne={len(shuffled)}"
        )
    return shuffled[:limit]


def split_counts(total: int, train_ratio: float, val_ratio: float):
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def load_class_sources(
    *,
    spec_entries: list[dict],
    class_name: str,
    spec_dir: Path,
    extensions: set[str],
    rng: random.Random,
):
    source_payloads = []
    for index, entry in enumerate(spec_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Zrodlo dla klasy {class_name} musi byc mapa.")

        source_name = str(entry.get("name") or f"{class_name}_source_{index}")
        source_path_value = entry.get("path")
        if not source_path_value:
            raise ValueError(f"Brak pola 'path' dla zrodla {source_name}.")

        source_path = resolve_root_path(spec_dir, str(source_path_value))
        if not source_path.exists():
            raise FileNotFoundError(f"Nie znaleziono zrodla {source_name}: {source_path}")

        limit_value = entry.get("limit")
        limit = int(limit_value) if limit_value is not None else None
        files = collect_files(source_path, extensions)
        if not files:
            raise ValueError(f"Brak obrazow w zrodle {source_name}: {source_path}")

        selected_files = select_subset(files, limit, label=f"{class_name}/{source_name}", rng=rng)
        source_payloads.append(
            {
                "name": source_name,
                "path": source_path,
                "files": selected_files,
            }
        )

    return source_payloads


def export_class_dataset(
    *,
    source_payloads: list[dict],
    output_dir: Path,
    class_name: str,
    train_ratio: float,
    val_ratio: float,
    copy: bool,
):
    manifest_entries = []
    split_summary = {
        "train": 0,
        "val": 0,
        "test": 0,
        "sources": {},
    }

    for source_payload in source_payloads:
        source_name = source_payload["name"]
        files = list(source_payload["files"])
        train_count, val_count, _ = split_counts(len(files), train_ratio, val_ratio)
        split_map = {
            "train": files[:train_count],
            "val": files[train_count : train_count + val_count],
            "test": files[train_count + val_count :],
        }

        split_summary["sources"][source_name] = {
            split_name: len(split_files) for split_name, split_files in split_map.items()
        }

        for split_name, split_files in split_map.items():
            split_summary[split_name] += len(split_files)
            for index, source in enumerate(split_files, start=1):
                destination = (
                    output_dir
                    / split_name
                    / class_name
                    / f"{source_name}__{index:05d}{source.suffix.lower()}"
                )
                copy_or_move(source, destination, copy=copy)
                manifest_entries.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "source_name": source_name,
                        "source_path": str(source),
                        "output_path": str(destination),
                    }
                )

    return manifest_entries, split_summary


def main():
    args = parse_args()
    spec = load_config(args.spec)
    spec_dir = args.spec.parent.resolve()

    settings = spec.get("settings", {})
    train_ratio = float(settings.get("train_ratio", 0.7))
    val_ratio = float(settings.get("val_ratio", 0.15))
    test_ratio = float(settings.get("test_ratio", 0.15))
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Suma train_ratio + val_ratio + test_ratio musi wynosic 1.0.")

    output_dir_value = args.output_dir or settings.get("output_dir", "data/face_adaptation")
    output_dir = resolve_root_path(spec_dir, str(output_dir_value))
    seed = int(args.seed if args.seed is not None else settings.get("seed", 42))
    extensions_value = settings.get("extensions", sorted(IMAGE_EXTENSIONS))
    if isinstance(extensions_value, str):
        extensions = {ext.strip().lower() for ext in extensions_value.split(",") if ext.strip()}
    else:
        extensions = {str(ext).strip().lower() for ext in extensions_value if str(ext).strip()}

    sources = spec.get("sources", {})
    fake_entries = list(sources.get("fake", []))
    real_entries = list(sources.get("real", []))
    if not fake_entries or not real_entries:
        raise ValueError("Specyfikacja musi zawierac zrodla dla sources.fake i sources.real.")

    rng = random.Random(seed)
    fake_sources = load_class_sources(
        spec_entries=fake_entries,
        class_name="fake",
        spec_dir=spec_dir,
        extensions=extensions,
        rng=rng,
    )
    real_sources = load_class_sources(
        spec_entries=real_entries,
        class_name="real",
        spec_dir=spec_dir,
        extensions=extensions,
        rng=rng,
    )

    manifest_entries = []
    fake_manifest, fake_summary = export_class_dataset(
        source_payloads=fake_sources,
        output_dir=output_dir,
        class_name="fake",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        copy=args.copy,
    )
    manifest_entries.extend(fake_manifest)

    real_manifest, real_summary = export_class_dataset(
        source_payloads=real_sources,
        output_dir=output_dir,
        class_name="real",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        copy=args.copy,
    )
    manifest_entries.extend(real_manifest)

    summary = {
        "output_dir": str(output_dir),
        "copy_mode": bool(args.copy),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "train": {"fake": fake_summary["train"], "real": real_summary["train"]},
            "val": {"fake": fake_summary["val"], "real": real_summary["val"]},
            "test": {"fake": fake_summary["test"], "real": real_summary["test"]},
        },
        "source_breakdown": {
            "fake": fake_summary["sources"],
            "real": real_summary["sources"],
        },
        "entries": manifest_entries,
    }

    summary_path = output_dir / "adaptation_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Zapisano dataset adaptacyjny modelu twarzowego: {output_dir}")
    print(f"Manifest: {summary_path}")
    for split_name in ["train", "val", "test"]:
        counts = summary["counts"][split_name]
        print(f"{split_name}: fake={counts['fake']}, real={counts['real']}")


if __name__ == "__main__":
    main()
