from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs into one CSV table.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/experiment_summary.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(run_dir: Path) -> dict[str, str | int | float]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "summary.json")

    row: dict[str, str | int | float] = {
        "run_name": run_dir.name,
        "arch": config.get("arch", ""),
        "weights": config.get("weights", ""),
        "epochs": config.get("epochs", ""),
        "batch_size": config.get("batch_size", ""),
        "learning_rate": config.get("learning_rate", ""),
        "weight_decay": config.get("weight_decay", ""),
        "image_size": config.get("image_size", ""),
        "train_size": config.get("train_size", ""),
        "val_size": config.get("val_size", ""),
        "augmentation_enabled": config.get("augmentation_enabled", ""),
        "device": config.get("device", ""),
        "best_val_accuracy": summary.get("best_val_accuracy", ""),
        "total_training_time": summary.get("total_training_time", ""),
        "best_checkpoint": summary.get("best_checkpoint", ""),
    }

    train_distribution = config.get("train_distribution", {})
    val_distribution = config.get("val_distribution", {})
    row["train_cat"] = train_distribution.get("cat", "")
    row["train_dog"] = train_distribution.get("dog", "")
    row["val_cat"] = val_distribution.get("cat", "")
    row["val_dog"] = val_distribution.get("dog", "")
    return row


def main() -> None:
    args = parse_args()
    run_dirs = sorted(
        run_dir
        for run_dir in args.outputs_dir.iterdir()
        if run_dir.is_dir()
        and (run_dir / "run_config.json").exists()
        and (run_dir / "summary.json").exists()
    )

    if not run_dirs:
        raise SystemExit(f"No completed runs found in {args.outputs_dir}")

    rows = [extract_row(run_dir) for run_dir in run_dirs]
    fieldnames = list(rows[0].keys())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summarized {len(rows)} runs into: {args.output_csv}")


if __name__ == "__main__":
    main()
