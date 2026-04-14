from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-10 runs.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/cifar10_experiment_summary.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dirs = [
        run_dir
        for run_dir in sorted(args.outputs_dir.iterdir())
        if run_dir.is_dir()
        and (run_dir / "run_config.json").exists()
        and (run_dir / "summary.json").exists()
        and "cifar" in run_dir.name.lower()
    ]
    if not run_dirs:
        raise SystemExit(f"No CIFAR-10 runs found in {args.outputs_dir}")

    rows = []
    for run_dir in run_dirs:
        config = load_json(run_dir / "run_config.json")
        summary = load_json(run_dir / "summary.json")
        row = {
            "run_name": run_dir.name,
            "arch": config.get("arch", ""),
            "weights": config.get("weights", ""),
            "epochs": config.get("epochs", ""),
            "batch_size": config.get("batch_size", ""),
            "learning_rate": config.get("learning_rate", ""),
            "train_size": config.get("train_size", ""),
            "test_size": config.get("test_size", ""),
            "augmentation_enabled": config.get("augmentation_enabled", ""),
            "imbalance": config.get("imbalance", ""),
            "imbalance_method": config.get("imbalance_method", ""),
            "majority_per_class": config.get("majority_per_class", ""),
            "minority_per_class": config.get("minority_per_class", ""),
            "minority_classes": config.get("minority_classes", ""),
            "best_test_accuracy": summary.get("best_test_accuracy", ""),
            "best_epoch": summary.get("best_epoch", ""),
            "total_training_time": summary.get("total_training_time", ""),
            "best_checkpoint": summary.get("best_checkpoint", ""),
        }
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summarized {len(rows)} CIFAR-10 runs into: {args.output_csv}")


if __name__ == "__main__":
    main()
