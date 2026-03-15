from pathlib import Path

from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool):
    transform_steps = [transforms.Resize((image_size, image_size))]
    if train:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_steps)


def load_datasets(data_dir: Path, image_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Brakuje katalogów danych: '{train_dir}', '{val_dir}' lub '{test_dir}'."
        )

    train_dataset = datasets.ImageFolder(
        root=str(train_dir), transform=build_transforms(image_size=image_size, train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir), transform=build_transforms(image_size=image_size, train=False)
    )
    test_dataset = datasets.ImageFolder(
        root=str(test_dir), transform=build_transforms(image_size=image_size, train=False)
    )
    return train_dataset, val_dataset, test_dataset
