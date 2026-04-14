from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import datasets, models, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class TestImageDataset(Dataset):
    def __init__(self, root: str | Path, transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(
            self.root.glob("*.jpg"),
            key=lambda path: int(path.stem),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(image_path.stem), image_path.name


@dataclass
class DatasetInfo:
    class_names: list[str]
    class_to_idx: dict[str, int]
    train_size: int
    val_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_transform(image_size: int, augmentation: bool = True) -> transforms.Compose:
    transform_steps: list[transforms.Compose | transforms.Resize | transforms.Normalize] = [
        transforms.Resize((image_size, image_size)),
    ]
    if augmentation:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(arch: str, num_classes: int, weights: str) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if arch not in {"resnet18", "resnet34"}:
        raise ValueError(f"Unsupported architecture: {arch}")

    if arch == "resnet18":
        model_fn = models.resnet18
        weight_enum = models.ResNet18_Weights.DEFAULT if weights == "imagenet" else None
    else:
        model_fn = models.resnet34
        weight_enum = models.ResNet34_Weights.DEFAULT if weights == "imagenet" else None

    try:
        model = model_fn(weights=weight_enum)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained weights. If you do not want to download them, "
            "rerun with '--weights none'."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _limit_subset_per_class(dataset: datasets.ImageFolder, per_class_limit: int) -> Subset:
    if per_class_limit <= 0:
        raise ValueError("per_class_limit must be a positive integer.")

    selected_indices: list[int] = []
    counts = {class_idx: 0 for class_idx in range(len(dataset.classes))}
    for sample_index, (_, class_idx) in enumerate(dataset.samples):
        if counts[class_idx] >= per_class_limit:
            continue
        selected_indices.append(sample_index)
        counts[class_idx] += 1
        if all(count >= per_class_limit for count in counts.values()):
            break
    return Subset(dataset, selected_indices)


def build_imagefolder_dataset(
    root: str | Path,
    transform: transforms.Compose,
    per_class_limit: int | None = None,
) -> datasets.ImageFolder | Subset:
    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    if per_class_limit is None:
        return dataset
    return _limit_subset_per_class(dataset, per_class_limit=per_class_limit)


def extract_dataset_info(dataset: datasets.ImageFolder | Subset) -> DatasetInfo:
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    return DatasetInfo(
        class_names=list(base_dataset.classes),
        class_to_idx=dict(base_dataset.class_to_idx),
        train_size=0,
        val_size=0,
    )


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_seconds(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def subset_length(dataset: datasets.ImageFolder | Subset) -> int:
    return len(dataset)


def get_base_dataset(dataset: datasets.ImageFolder | Subset) -> datasets.ImageFolder:
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def summarize_class_distribution(dataset: datasets.ImageFolder | Subset) -> dict[str, int]:
    base_dataset = get_base_dataset(dataset)
    index_iterable: Iterable[int]
    if isinstance(dataset, Subset):
        index_iterable = dataset.indices
    else:
        index_iterable = range(len(base_dataset.samples))

    counts = {class_name: 0 for class_name in base_dataset.classes}
    for sample_index in index_iterable:
        _, class_idx = base_dataset.samples[sample_index]
        class_name = base_dataset.classes[class_idx]
        counts[class_name] += 1
    return counts
