from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zloz kontrolowany multisource dataset twarzy do treningu ConvNeXt."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("curated_faces_v2_sources.yaml"),
        help="Plik YAML ze zrodlami real/fake i limitami na zrodlo.",
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


def load_spec(spec_path: Path):
    if not spec_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku specyfikacji: {spec_path}")
    with spec_path.open("r", encoding="utf-8") as spec_file:
        spec = yaml.safe_load(spec_file) or {}
    if not isinstance(spec, dict):
        raise ValueError("Specyfikacja YAML musi zawierac mape na poziomie root.")
    return spec


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


def select_subset(files: list[Path], limit: int | None, *, label: str):
    if limit is None:
        return list(files)
    if limit <= 0:
        raise ValueError(f"Limit dla {label} musi byc dodatni albo pusty.")
    if len(files) < limit:
        raise ValueError(
            f"Za malo plikow dla {label}: wymagane={limit}, dostepne={len(files)}"
        )
    return files[:limit]


def split_counts(total: int, train_ratio: float, val_ratio: float):
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_or_move(source: Path, destination: Path, copy: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def ensure_output_dirs(output_dir: Path):
    for split_name in ["train", "val", "test"]:
        for class_name in ["fake", "real"]:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)


def load_class_sources(
    *,
    spec_entries: list[dict],
    class_name: str,
    spec_dir: Path,
    extensions: set[str],
):
    source_payloads = []
    for entry in spec_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Zrodlo dla klasy {class_name} musi byc mapa.")
        source_name = str(entry.get("name") or f"{class_name}_source_{len(source_payloads) + 1}")
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

        selected_files = select_subset(files, limit, label=f"{class_name}/{source_name}")
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
    spec = load_spec(args.spec)
    spec_dir = args.spec.parent.resolve()

    settings = spec.get("settings", {})
    train_ratio = float(settings.get("train_ratio", 0.7))
    val_ratio = float(settings.get("val_ratio", 0.15))
    test_ratio = float(settings.get("test_ratio", 0.15))
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Suma train_ratio + val_ratio + test_ratio musi wynosic 1.0.")

    output_dir_value = args.output_dir or settings.get("output_dir", "data/deepfake_faces_v2")
    output_dir = resolve_root_path(spec_dir, str(output_dir_value))
    seed = int(args.seed if args.seed is not None else settings.get("seed", 42))
    extensions_value = settings.get("extensions", sorted(IMAGE_EXTENSIONS))
    if isinstance(extensions_value, str):
        extensions = {ext.strip().lower() for ext in extensions_value.split(",") if ext.strip()}
    else:
        extensions = {str(ext).strip().lower() for ext in extensions_value if str(ext).strip()}

    sources = spec.get("sources", {})
    real_entries = list(sources.get("real", []))
    fake_entries = list(sources.get("fake", []))
    if not real_entries or not fake_entries:
        raise ValueError("Specyfikacja musi zawierac zrodla dla sources.real i sources.fake.")

    random.seed(seed)
    ensure_output_dirs(output_dir)

    real_sources = load_class_sources(
        spec_entries=real_entries,
        class_name="real",
        spec_dir=spec_dir,
        extensions=extensions,
    )
    fake_sources = load_class_sources(
        spec_entries=fake_entries,
        class_name="fake",
        spec_dir=spec_dir,
        extensions=extensions,
    )

    for payload in real_sources:
        random.shuffle(payload["files"])
    for payload in fake_sources:
        random.shuffle(payload["files"])

    real_manifest, real_summary = export_class_dataset(
        source_payloads=real_sources,
        output_dir=output_dir,
        class_name="real",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        copy=args.copy,
    )
    fake_manifest, fake_summary = export_class_dataset(
        source_payloads=fake_sources,
        output_dir=output_dir,
        class_name="fake",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        copy=args.copy,
    )

    summary = {
        "spec": str(args.spec),
        "output_dir": str(output_dir),
        "copy_mode": bool(args.copy),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "class_summary": {
            "real": real_summary,
            "fake": fake_summary,
        },
        "entries": real_manifest + fake_manifest,
    }
    manifest_path = output_dir / "curated_dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(summary, manifest_file, indent=2)

    print(f"Zapisano curated dataset v2: {output_dir}")
    print(f"Manifest: {manifest_path}")
    for class_name, class_summary in summary["class_summary"].items():
        print(
            f"{class_name}: "
            f"train={class_summary['train']}, "
            f"val={class_summary['val']}, "
            f"test={class_summary['test']}"
        )


if __name__ == "__main__":
    main()
