# Submission Checklist

## Files already prepared

- Final checkpoint: `outputs/final_resnet34_full/best_model.pt`
- Final prediction file: `outputs/final_resnet34_full/submission.csv`
- Final training summary: `outputs/final_resnet34_full/summary.json`
- Final training history: `outputs/final_resnet34_full/history.csv`
- Final run configuration: `outputs/final_resnet34_full/run_config.json`

## What to include in the final report

1. Group member names and contributions.
2. Literature survey.
3. Dataset description and data split actually used.
4. Data preprocessing and augmentation.
5. Model architecture and training strategy.
6. Parameter choices and reasons.
7. Validation accuracy results and model comparison table.
8. Correct and incorrect sample analysis. See `report/error_sample_analysis.md`.
9. CIFAR-10 extension results. See `report/cifar10_final_results.md`.
10. Data imbalance handling methods for CIFAR-10. See `report/cifar10_final_results.md`.
11. References.

## Dogs vs Cats numbers you can directly use

- Training images: `20,000`
- Validation images: `5,000`
- Final model: `ResNet34`
- Pretrained: `Yes`
- Best validation accuracy: `99.02%`
- Best epoch: `2`
- Total training time: `10:05`
- Label encoding: `cat = 0`, `dog = 1`

## Important checks before submission

1. Open `outputs/final_resnet34_full/submission.csv` and confirm the columns are exactly `id,label`.
2. Confirm the report clearly states that `cat = 0` and `dog = 1`.
3. Confirm the report uses the best checkpoint result rather than the final epoch result.
4. Confirm all figures and tables match the numbers in `history.csv` and `summary.json`.
5. Confirm references are cited properly.
