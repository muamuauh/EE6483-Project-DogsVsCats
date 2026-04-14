from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_utils import TestImageDataset, build_eval_transform, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv for Dogs vs Cats.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("datasets/datasets/test"))
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    model = build_model(
        arch=checkpoint["arch"],
        num_classes=len(class_to_idx),
        weights="none",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = TestImageDataset(
        root=args.test_dir,
        transform=build_eval_transform(image_size=checkpoint["image_size"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: list[tuple[int, int]] = []
    with torch.no_grad():
        for images, image_ids, _ in test_loader:
            logits = model(images.to(device))
            predicted_labels = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(zip(image_ids.tolist(), predicted_labels))

    predictions.sort(key=lambda row: row[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
