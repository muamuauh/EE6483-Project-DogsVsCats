# CIFAR-10 Experiment Pack

This pack covers assignment parts (g) and (h):

- (g) Apply and improve the classifier for CIFAR-10 multi-category image classification.
- (h) Train with class imbalance and justify at least two approaches.

## CIFAR-10 class ids

| id | class |
|---:|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Quick smoke test

Run this first to download CIFAR-10 and verify the script works:

```powershell
python train_cifar10.py `
  --data-dir datasets/cifar10 `
  --output-dir outputs/cifar10_smoke `
  --arch resnet18 `
  --weights imagenet `
  --epochs 1 `
  --batch-size 32 `
  --train-per-class 50 `
  --test-per-class 20 `
  --download `
  --device cuda
```

## Part (g): Balanced CIFAR-10 baseline

This run uses the normal CIFAR-10 training set and official test set.

```powershell
python train_cifar10.py `
  --data-dir datasets/cifar10 `
  --output-dir outputs/cifar10_resnet18_balanced `
  --arch resnet18 `
  --weights imagenet `
  --epochs 10 `
  --batch-size 64 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --download `
  --device cuda
```

If GPU memory is tight, change `--batch-size 64` to `--batch-size 32`.

## Part (h): Imbalanced CIFAR-10 baseline

This creates an imbalanced training set where classes 5 to 9 have fewer labelled samples:

- majority classes 0 to 4: `5000` images per class
- minority classes 5 to 9: `500` images per class

No imbalance correction:

```powershell
python train_cifar10.py `
  --data-dir datasets/cifar10 `
  --output-dir outputs/cifar10_imb_none `
  --arch resnet18 `
  --weights imagenet `
  --epochs 10 `
  --batch-size 64 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --imbalance `
  --majority-per-class 5000 `
  --minority-per-class 500 `
  --minority-classes 5,6,7,8,9 `
  --imbalance-method none `
  --download `
  --device cuda
```

Approach 1, class-weighted loss:

```powershell
python train_cifar10.py `
  --data-dir datasets/cifar10 `
  --output-dir outputs/cifar10_imb_weighted_loss `
  --arch resnet18 `
  --weights imagenet `
  --epochs 10 `
  --batch-size 64 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --imbalance `
  --majority-per-class 5000 `
  --minority-per-class 500 `
  --minority-classes 5,6,7,8,9 `
  --imbalance-method weighted_loss `
  --download `
  --device cuda
```

Approach 2, oversampling:

```powershell
python train_cifar10.py `
  --data-dir datasets/cifar10 `
  --output-dir outputs/cifar10_imb_oversampling `
  --arch resnet18 `
  --weights imagenet `
  --epochs 10 `
  --batch-size 64 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --imbalance `
  --majority-per-class 5000 `
  --minority-per-class 500 `
  --minority-classes 5,6,7,8,9 `
  --imbalance-method oversampling `
  --download `
  --device cuda
```

## Summarize runs

```powershell
python summarize_cifar_runs.py `
  --outputs-dir outputs `
  --output-csv outputs/cifar10_experiment_summary.csv
```

## What to report

For part (g):

- CIFAR-10 has 10 classes of `32 x 32` color images.
- The model output layer was changed from 2 classes to 10 classes.
- The same transfer-learning idea was reused from Dogs vs Cats.
- Report test accuracy from `summary.json`.

For part (h):

- Explain the constructed imbalance.
- Compare `none`, `weighted_loss`, and `oversampling`.
- Use `best_per_class_accuracy.json` to discuss minority-class performance.
