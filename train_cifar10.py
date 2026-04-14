from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from baseline_utils import build_model, format_seconds, resolve_device, set_seed


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier and imbalance variants.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cifar10_resnet18"))
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "simple_cnn"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--imbalance", action="store_true")
    parser.add_argument("--majority-per-class", type=int, default=5000)
    parser.add_argument("--minority-per-class", type=int, default=500)
    parser.add_argument("--minority-classes", type=str, default="5,6,7,8,9")
    parser.add_argument("--imbalance-method", choices=["none", "weighted_loss", "oversampling"], default="none")
    return parser.parse_args()


def build_train_transform(image_size: int, augmentation: bool) -> transforms.Compose:
    steps: list = []
    if augmentation:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_class_ids(class_id_text: str) -> set[int]:
    if not class_id_text.strip():
        return set()
    class_ids = {int(item.strip()) for item in class_id_text.split(",") if item.strip()}
    invalid = sorted(class_id for class_id in class_ids if class_id < 0 or class_id >= len(CIFAR10_CLASSES))
    if invalid:
        raise ValueError(f"Invalid CIFAR-10 class ids: {invalid}")
    return class_ids


def targets_of(dataset: datasets.CIFAR10 | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = dataset.dataset.targets
        return [int(base_targets[index]) for index in dataset.indices]
    return [int(target) for target in dataset.targets]


def class_distribution(dataset: datasets.CIFAR10 | Subset) -> dict[str, int]:
    counts = Counter(targets_of(dataset))
    return {CIFAR10_CLASSES[class_id]: counts.get(class_id, 0) for class_id in range(len(CIFAR10_CLASSES))}


def make_limited_subset(
    dataset: datasets.CIFAR10,
    per_class: int | None,
    seed: int,
) -> datasets.CIFAR10 | Subset:
    if per_class is None:
        return dataset
    return make_count_subset(
        dataset=dataset,
        per_class_counts={class_id: per_class for class_id in range(len(CIFAR10_CLASSES))},
        seed=seed,
    )


def make_imbalanced_subset(
    dataset: datasets.CIFAR10,
    majority_per_class: int,
    minority_per_class: int,
    minority_classes: set[int],
    seed: int,
) -> Subset:
    per_class_counts = {}
    for class_id in range(len(CIFAR10_CLASSES)):
        per_class_counts[class_id] = minority_per_class if class_id in minority_classes else majority_per_class
    return make_count_subset(dataset=dataset, per_class_counts=per_class_counts, seed=seed)


def make_count_subset(
    dataset: datasets.CIFAR10,
    per_class_counts: dict[int, int],
    seed: int,
) -> Subset:
    rng = random.Random(seed)
    class_to_indices: dict[int, list[int]] = {class_id: [] for class_id in range(len(CIFAR10_CLASSES))}
    for sample_index, target in enumerate(dataset.targets):
        class_to_indices[int(target)].append(sample_index)

    selected_indices: list[int] = []
    for class_id, requested_count in per_class_counts.items():
        indices = list(class_to_indices[class_id])
        rng.shuffle(indices)
        selected_indices.extend(indices[: min(requested_count, len(indices))])

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def build_sampler(dataset: datasets.CIFAR10 | Subset, method: str) -> WeightedRandomSampler | None:
    if method != "oversampling":
        return None
    targets = targets_of(dataset)
    counts = Counter(targets)
    sample_weights = [1.0 / counts[target] for target in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_criterion(dataset: datasets.CIFAR10 | Subset, method: str, device: torch.device) -> nn.Module:
    if method != "weighted_loss":
        return nn.CrossEntropyLoss()
    targets = targets_of(dataset)
    counts = Counter(targets)
    total = len(targets)
    weights = []
    for class_id in range(len(CIFAR10_CLASSES)):
        class_count = counts.get(class_id, 0)
        weights.append(total / (len(CIFAR10_CLASSES) * class_count) if class_count else 0.0)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_class_correct = Counter()
    per_class_total = Counter()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

            for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                per_class_total[label] += 1
                if label == prediction:
                    per_class_correct[label] += 1

    per_class_accuracy = {
        CIFAR10_CLASSES[class_id]: (
            per_class_correct[class_id] / per_class_total[class_id] if per_class_total[class_id] else 0.0
        )
        for class_id in range(len(CIFAR10_CLASSES))
    }
    return total_loss / total_samples, total_correct / total_samples, per_class_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_base = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=build_train_transform(args.image_size, augmentation=not args.disable_augmentation),
    )
    test_dataset_base = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=args.download,
        transform=build_eval_transform(args.image_size),
    )

    if args.imbalance:
        train_dataset = make_imbalanced_subset(
            train_dataset_base,
            majority_per_class=args.majority_per_class,
            minority_per_class=args.minority_per_class,
            minority_classes=parse_class_ids(args.minority_classes),
            seed=args.seed,
        )
    else:
        train_dataset = make_limited_subset(train_dataset_base, per_class=args.train_per_class, seed=args.seed)

    test_dataset = make_limited_subset(test_dataset_base, per_class=args.test_per_class, seed=args.seed)
    sampler = build_sampler(train_dataset, args.imbalance_method)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.arch, num_classes=len(CIFAR10_CLASSES), weights=args.weights).to(device)
    criterion = build_criterion(train_dataset, args.imbalance_method, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "arch": args.arch,
        "weights": args.weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "seed": args.seed,
        "download": args.download,
        "augmentation_enabled": not args.disable_augmentation,
        "imbalance": args.imbalance,
        "imbalance_method": args.imbalance_method,
        "majority_per_class": args.majority_per_class if args.imbalance else None,
        "minority_per_class": args.minority_per_class if args.imbalance else None,
        "minority_classes": args.minority_classes if args.imbalance else None,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "train_distribution": class_distribution(train_dataset),
        "test_distribution": class_distribution(test_dataset),
        "classes": CIFAR10_CLASSES,
    }
    save_json(config, args.output_dir / "run_config.json")

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train distribution: {class_distribution(train_dataset)}")
    print(f"Imbalance method: {args.imbalance_method}")

    history_rows = []
    best_test_accuracy = -1.0
    best_epoch = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, per_class_accuracy = evaluate(model, test_loader, criterion, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "epoch_seconds": time.time() - epoch_start,
        }
        history_rows.append(history_row)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f} | "
            f"time={format_seconds(history_row['epoch_seconds'])}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "weights": args.weights,
            "image_size": args.image_size,
            "classes": CIFAR10_CLASSES,
            "best_test_accuracy": max(best_test_accuracy, test_accuracy),
            "epoch": epoch,
        }
        torch.save(checkpoint, args.output_dir / "last_model.pt")
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(checkpoint, args.output_dir / "best_model.pt")
            save_json(per_class_accuracy, args.output_dir / "best_per_class_accuracy.json")
            print(f"Saved new best checkpoint with test_acc={best_test_accuracy:.4f}")

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
        "epochs_completed": args.epochs,
        "total_training_seconds": time.time() - start,
        "total_training_time": format_seconds(time.time() - start),
        "best_checkpoint": str(args.output_dir / "best_model.pt"),
        "last_checkpoint": str(args.output_dir / "last_model.pt"),
    }
    save_json(summary, args.output_dir / "summary.json")

    print()
    print("CIFAR-10 training finished.")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model saved to: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
