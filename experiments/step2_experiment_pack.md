# Step 2 Experiment Pack

This file turns the baseline into a report-ready experiment set for the `Dogs vs Cats` task.

## Goal of step 2

Run a small number of controlled experiments so you can answer these report questions later:

- what baseline model you chose
- how preprocessing and augmentation affect validation accuracy
- how different models compare
- how your parameter choices were decided

## Recommended experiment set

Run these in order. Start with the quicker subset first, then scale up if your machine is comfortable.

### Experiment A: Main baseline

Purpose: your primary `Dogs vs Cats` baseline for the report.

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/exp_a_resnet18_baseline `
  --arch resnet18 `
  --weights imagenet `
  --epochs 5 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --train-per-class 2000 `
  --val-per-class 500
```

If pretrained weights cannot be downloaded, switch `--weights imagenet` to `--weights none`.

### Experiment B: No augmentation

Purpose: isolate the effect of augmentation.

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/exp_b_resnet18_no_aug `
  --arch resnet18 `
  --weights imagenet `
  --epochs 5 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --train-per-class 2000 `
  --val-per-class 500 `
  --disable-augmentation
```

### Experiment C: Simple CNN comparison

Purpose: compare a custom CNN against ResNet.

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/exp_c_simple_cnn `
  --arch simple_cnn `
  --weights none `
  --epochs 5 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --train-per-class 2000 `
  --val-per-class 500
```

### Experiment D: Deeper ResNet comparison

Purpose: compare `ResNet18` and `ResNet34`.

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/exp_d_resnet34 `
  --arch resnet34 `
  --weights imagenet `
  --epochs 5 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --train-per-class 2000 `
  --val-per-class 500
```

### Experiment E: Longer training on your best setup

Purpose: create a stronger final model for `submission.csv`.

Pick the best result from A to D, then scale it up:

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/exp_e_best_longer `
  --arch resnet18 `
  --weights imagenet `
  --epochs 8 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --train-per-class 5000 `
  --val-per-class 1000
```

If your machine is strong enough, you can remove `--train-per-class` and `--val-per-class` entirely to use all available data.

## After each training run

Generate the corresponding prediction file:

```powershell
python predict.py `
  --checkpoint outputs/exp_a_resnet18_baseline/best_model.pt `
  --test-dir datasets/datasets/test `
  --output-csv outputs/exp_a_resnet18_baseline/submission.csv
```

Change the output directory to match the run you want.

## Summarize all completed experiments

After several runs, generate one comparison table:

```powershell
python summarize_runs.py `
  --outputs-dir outputs `
  --output-csv outputs/experiment_summary.csv
```

## What to look for

- `best_val_accuracy` in each run's `summary.json`
- training and validation trends in `history.csv`
- whether augmentation improves validation accuracy
- whether pretrained `ResNet18` beats a simple CNN
- whether a deeper model gives enough gain to justify more computation
