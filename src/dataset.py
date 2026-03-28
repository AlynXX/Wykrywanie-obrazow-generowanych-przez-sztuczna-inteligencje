from __future__ import annotations

import io
import random
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomJPEGCompression:
    def __init__(self, min_quality: int = 45, max_quality: int = 95, p: float = 0.6):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return compressed.copy()


class AddGaussianNoise:
    def __init__(self, std_range: tuple[float, float] = (0.01, 0.05), p: float = 0.4):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def robust_pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as image_file:
        image = Image.open(image_file)
        return image.convert("RGB")


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned_paths: set[str] = set()

    def __getitem__(self, index: int):
        max_attempts = min(16, len(self.samples))
        current_index = index

        for _ in range(max_attempts):
            path, _ = self.samples[current_index]
            try:
                return super().__getitem__(current_index)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                if path not in self._warned_paths:
                    warnings.warn(
                        f"Pomijam uszkodzony obraz: {path} ({error})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_paths.add(path)
                current_index = (current_index + 1) % len(self.samples)

        raise RuntimeError(
            "Nie udalo sie wczytac poprawnego obrazu po wielu probach. "
            "Sprawdz, czy dataset nie zawiera zbyt wielu uszkodzonych plikow."
        )


def build_transforms(image_size: int, train: bool):
    transform_steps = [transforms.Resize((image_size, image_size))]
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandomJPEGCompression(p=0.6),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                    p=0.3,
                ),
            ]
        )

    transform_steps.append(transforms.ToTensor())

    if train:
        transform_steps.append(AddGaussianNoise(std_range=(0.01, 0.04), p=0.4))

    transform_steps.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_steps)


def load_datasets(data_dir: Path, image_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Brakuje katalogow danych: '{train_dir}', '{val_dir}' lub '{test_dir}'."
        )

    train_dataset = SafeImageFolder(
        root=str(train_dir),
        transform=build_transforms(image_size=image_size, train=True),
        loader=robust_pil_loader,
    )
    val_dataset = SafeImageFolder(
        root=str(val_dir),
        transform=build_transforms(image_size=image_size, train=False),
        loader=robust_pil_loader,
    )
    test_dataset = SafeImageFolder(
        root=str(test_dir),
        transform=build_transforms(image_size=image_size, train=False),
        loader=robust_pil_loader,
    )
    return train_dataset, val_dataset, test_dataset
