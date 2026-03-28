from __future__ import annotations

import io
import random
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resampling(name: str):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


class RandomJPEGCompression:
    def __init__(self, min_quality: int = 25, max_quality: int = 95, p: float = 0.7):
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


class RandomDownscaleUpscale:
    def __init__(self, min_scale: float = 0.35, max_scale: float = 0.85, p: float = 0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p
        self.downscale_resampling = (
            _resampling("NEAREST"),
            _resampling("BILINEAR"),
            _resampling("BICUBIC"),
        )
        self.upscale_resampling = (
            _resampling("BILINEAR"),
            _resampling("BICUBIC"),
            _resampling("LANCZOS"),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        width, height = image.size
        if width < 2 or height < 2:
            return image

        scale = random.uniform(self.min_scale, self.max_scale)
        reduced_width = max(24, int(width * scale))
        reduced_height = max(24, int(height * scale))
        reduced_width = min(reduced_width, width)
        reduced_height = min(reduced_height, height)

        downsampled = image.resize(
            (reduced_width, reduced_height),
            resample=random.choice(self.downscale_resampling),
        )
        restored = downsampled.resize(
            (width, height),
            resample=random.choice(self.upscale_resampling),
        )
        return restored


class AddGaussianNoise:
    def __init__(self, std_range: tuple[float, float] = (0.01, 0.06), p: float = 0.45):
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


def build_quality_profile_steps(quality_profile: str):
    profile = quality_profile.strip().lower()
    if profile == "clean":
        return []
    if profile == "jpeg_low":
        return [RandomJPEGCompression(min_quality=25, max_quality=35, p=1.0)]
    if profile == "jpeg_extreme":
        return [RandomJPEGCompression(min_quality=10, max_quality=20, p=1.0)]
    if profile == "blur_light":
        return [transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 0.8))]
    if profile == "blur_strong":
        return [transforms.GaussianBlur(kernel_size=7, sigma=(1.8, 1.8))]
    if profile == "downscale_light":
        return [RandomDownscaleUpscale(min_scale=0.55, max_scale=0.55, p=1.0)]
    if profile == "downscale_strong":
        return [RandomDownscaleUpscale(min_scale=0.35, max_scale=0.35, p=1.0)]
    if profile == "mixed_quality":
        return [
            RandomDownscaleUpscale(min_scale=0.4, max_scale=0.55, p=1.0),
            RandomJPEGCompression(min_quality=12, max_quality=28, p=1.0),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.1, 1.1)),
        ]

    raise ValueError(
        "Nieznany profil jakosci obrazu: "
        f"{quality_profile}. Dostepne: {', '.join(list_quality_profiles())}"
    )


def list_quality_profiles():
    return [
        "clean",
        "jpeg_low",
        "jpeg_extreme",
        "blur_light",
        "blur_strong",
        "downscale_light",
        "downscale_strong",
        "mixed_quality",
    ]


def build_transforms(image_size: int, train: bool, quality_profile: str | None = None):
    transform_steps = [transforms.Resize((image_size, image_size))]
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandomDownscaleUpscale(p=0.5),
                RandomJPEGCompression(p=0.7),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 2.2))],
                    p=0.4,
                ),
            ]
        )
    elif quality_profile is not None:
        transform_steps.extend(build_quality_profile_steps(quality_profile))

    transform_steps.append(transforms.ToTensor())

    if train:
        transform_steps.append(AddGaussianNoise(std_range=(0.01, 0.06), p=0.45))

    transform_steps.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_steps)


def create_image_folder_dataset(
    root: Path,
    image_size: int,
    train: bool,
    quality_profile: str | None = None,
):
    return SafeImageFolder(
        root=str(root),
        transform=build_transforms(
            image_size=image_size,
            train=train,
            quality_profile=quality_profile,
        ),
        loader=robust_pil_loader,
    )


def load_datasets(data_dir: Path, image_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Brakuje katalogow danych: '{train_dir}', '{val_dir}' lub '{test_dir}'."
        )

    train_dataset = create_image_folder_dataset(root=train_dir, image_size=image_size, train=True)
    val_dataset = create_image_folder_dataset(root=val_dir, image_size=image_size, train=False)
    test_dataset = create_image_folder_dataset(root=test_dir, image_size=image_size, train=False)
    return train_dataset, val_dataset, test_dataset
