from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets

from baseline_utils import build_eval_transform, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze correct and incorrect validation samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=Path("datasets/datasets/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_resnet34_full/error_analysis"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def make_contact_sheet(rows: list[dict], output_path: Path, title: str, thumb_size: int = 180) -> None:
    if not rows:
        return

    cols = min(3, len(rows))
    caption_height = 62
    title_height = 38
    rows_count = (len(rows) + cols - 1) // cols
    sheet_width = cols * thumb_size
    sheet_height = title_height + rows_count * (thumb_size + caption_height)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill="black")

    for idx, row in enumerate(rows):
        grid_x = idx % cols
        grid_y = idx // cols
        x = grid_x * thumb_size
        y = title_height + grid_y * (thumb_size + caption_height)

        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        paste_x = x + (thumb_size - image.width) // 2
        paste_y = y + (thumb_size - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))

        caption_y = y + thumb_size + 4
        caption = (
            f"{Path(row['image_path']).name}\n"
            f"T:{row['true_label']} P:{row['pred_label']}\n"
            f"conf={row['confidence']:.3f}"
        )
        draw.text((x + 6, caption_y), caption, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_markdown(
    output_path: Path,
    accuracy: float,
    total: int,
    correct: int,
    incorrect: int,
    selected_groups: dict[str, list[dict]],
) -> None:
    lines = [
        "# Validation Error Analysis",
        "",
        f"- Accuracy: `{accuracy:.4f}`",
        f"- Correct samples: `{correct}`",
        f"- Incorrect samples: `{incorrect}`",
        f"- Total validation samples: `{total}`",
        "",
    ]

    for title, rows in selected_groups.items():
        lines.extend([f"## {title}", "", "| image | true | predicted | confidence |", "|---|---|---|---|"])
        for row in rows:
            lines.append(
                f"| `{Path(row['image_path']).name}` | `{row['true_label']}` | "
                f"`{row['pred_label']}` | `{row['confidence']:.4f}` |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def copy_selected(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        source = Path(row["image_path"])
        new_name = (
            f"{idx:02d}_true-{row['true_label']}_pred-{row['pred_label']}"
            f"_conf-{row['confidence']:.3f}_{source.name}"
        )
        shutil.copy2(source, output_dir / new_name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    dataset = datasets.ImageFolder(
        root=str(args.val_dir),
        transform=build_eval_transform(image_size=checkpoint["image_size"]),
    )
    class_names = list(dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        arch=checkpoint["arch"],
        num_classes=len(class_names),
        weights="none",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows: list[dict] = []
    sample_index = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for batch_idx in range(images.size(0)):
                image_path, _ = dataset.samples[sample_index]
                true_idx = int(labels[batch_idx].item())
                pred_idx = int(predictions[batch_idx].item())
                row = {
                    "image_path": str(Path(image_path)),
                    "true_label": class_names[true_idx],
                    "pred_label": class_names[pred_idx],
                    "correct": true_idx == pred_idx,
                    "confidence": float(confidences[batch_idx].item()),
                    "prob_cat": float(probabilities[batch_idx, dataset.class_to_idx["cat"]].item()),
                    "prob_dog": float(probabilities[batch_idx, dataset.class_to_idx["dog"]].item()),
                }
                rows.append(row)
                sample_index += 1

    correct_rows = [row for row in rows if row["correct"]]
    incorrect_rows = [row for row in rows if not row["correct"]]
    high_conf_incorrect = sorted(incorrect_rows, key=lambda row: row["confidence"], reverse=True)[: args.top_k]
    high_conf_correct = sorted(correct_rows, key=lambda row: row["confidence"], reverse=True)[: args.top_k]
    low_conf_correct = sorted(correct_rows, key=lambda row: row["confidence"])[: args.top_k]

    confusion = []
    for true_label in class_names:
        for pred_label in class_names:
            count = sum(1 for row in rows if row["true_label"] == true_label and row["pred_label"] == pred_label)
            confusion.append({"true_label": true_label, "pred_label": pred_label, "count": count})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        rows,
        args.output_dir / "val_predictions.csv",
        ["image_path", "true_label", "pred_label", "correct", "confidence", "prob_cat", "prob_dog"],
    )
    write_csv(confusion, args.output_dir / "confusion_matrix.csv", ["true_label", "pred_label", "count"])
    write_selected_markdown(
        args.output_dir / "selected_samples.md",
        accuracy=len(correct_rows) / len(rows),
        total=len(rows),
        correct=len(correct_rows),
        incorrect=len(incorrect_rows),
        selected_groups={
            "High-confidence incorrect samples": high_conf_incorrect,
            "High-confidence correct samples": high_conf_correct,
            "Low-confidence correct samples": low_conf_correct,
        },
    )

    copy_selected(high_conf_incorrect, args.output_dir / "incorrect_samples")
    copy_selected(high_conf_correct, args.output_dir / "correct_samples")
    copy_selected(low_conf_correct, args.output_dir / "low_confidence_correct_samples")
    make_contact_sheet(high_conf_incorrect, args.output_dir / "incorrect_contact_sheet.jpg", "High-confidence incorrect")
    make_contact_sheet(high_conf_correct, args.output_dir / "correct_contact_sheet.jpg", "High-confidence correct")
    make_contact_sheet(
        low_conf_correct,
        args.output_dir / "low_confidence_correct_contact_sheet.jpg",
        "Low-confidence correct",
    )

    print(f"Validation samples: {len(rows)}")
    print(f"Correct: {len(correct_rows)}")
    print(f"Incorrect: {len(incorrect_rows)}")
    print(f"Accuracy: {len(correct_rows) / len(rows):.4f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
