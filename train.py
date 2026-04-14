from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline_utils import (
    build_imagefolder_dataset,
    build_model,
    build_train_transform,
    build_eval_transform,
    count_parameters,
    format_seconds,
    resolve_device,
    save_json,
    set_seed,
    subset_length,
    summarize_class_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Dogs vs Cats baseline classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18_baseline"))
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "simple_cnn"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)
    parser.add_argument("--disable-augmentation", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    return epoch_loss, epoch_accuracy


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    best_val_accuracy: float,
    epoch: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "arch": args.arch,
        "weights": args.weights,
        "image_size": args.image_size,
        "class_to_idx": class_to_idx,
        "best_val_accuracy": best_val_accuracy,
        "epoch": epoch,
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_imagefolder_dataset(
        root=train_dir,
        transform=build_train_transform(
            image_size=args.image_size,
            augmentation=not args.disable_augmentation,
        ),
        per_class_limit=args.train_per_class,
    )
    val_dataset = build_imagefolder_dataset(
        root=val_dir,
        transform=build_eval_transform(image_size=args.image_size),
        per_class_limit=args.val_per_class,
    )

    base_train_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    class_to_idx = dict(base_train_dataset.class_to_idx)
    class_names = list(base_train_dataset.classes)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(arch=args.arch, num_classes=len(class_names), weights=args.weights).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config_payload = {
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
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "augmentation_enabled": not args.disable_augmentation,
        "train_size": subset_length(train_dataset),
        "val_size": subset_length(val_dataset),
        "train_distribution": summarize_class_distribution(train_dataset),
        "val_distribution": summarize_class_distribution(val_dataset),
        "class_to_idx": class_to_idx,
        "parameter_count": count_parameters(model),
    }
    save_json(config_payload, args.output_dir / "run_config.json")

    history_rows: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0

    print(f"Using device: {device}")
    print(f"Training samples: {subset_length(train_dataset)}")
    print(f"Validation samples: {subset_length(val_dataset)}")
    print(f"Train class distribution: {summarize_class_distribution(train_dataset)}")
    print(f"Val class distribution: {summarize_class_distribution(val_dataset)}")
    print(f"Model: {args.arch} ({count_parameters(model):,} trainable parameters)")

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
            )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": time.time() - epoch_start,
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} | "
            f"time={format_seconds(history_row['epoch_seconds'])}"
        )

        save_checkpoint(
            output_path=args.output_dir / "last_model.pt",
            model=model,
            args=args,
            class_to_idx=class_to_idx,
            best_val_accuracy=max(best_val_accuracy, val_accuracy),
            epoch=epoch,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                output_path=args.output_dir / "best_model.pt",
                model=model,
                args=args,
                class_to_idx=class_to_idx,
                best_val_accuracy=best_val_accuracy,
                epoch=epoch,
            )
            print(f"Saved new best checkpoint with val_acc={best_val_accuracy:.4f}")

    total_training_seconds = time.time() - training_start

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary_payload = {
        "best_val_accuracy": best_val_accuracy,
        "epochs_completed": args.epochs,
        "total_training_seconds": total_training_seconds,
        "total_training_time": format_seconds(total_training_seconds),
        "best_checkpoint": str(args.output_dir / "best_model.pt"),
        "last_checkpoint": str(args.output_dir / "last_model.pt"),
    }
    save_json(summary_payload, args.output_dir / "summary.json")

    print()
    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"History saved to: {history_path}")
    print(f"Best model saved to: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
