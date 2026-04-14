# EE6483 Mini Project Option 2: Dogs vs Cats and CIFAR-10

This repository contains the code, data, experiment outputs, result summaries, and report materials for EE6483 Artificial Intelligence and Data Mining Mini Project Option 2.

The project has two main parts:

- Dogs vs Cats binary image classification.
- CIFAR-10 multi-class classification and class imbalance experiments.

## Repository Structure

```text
.
├── train.py                         # Dogs vs Cats training
├── predict.py                       # Dogs vs Cats test prediction and submission.csv generation
├── baseline_utils.py                # Shared model, dataset, and transform utilities
├── analyze_val_errors.py            # Dogs vs Cats validation error analysis
├── train_cifar10.py                 # CIFAR-10 balanced and imbalanced training
├── summarize_runs.py                # Dogs vs Cats run summarizer
├── summarize_cifar_runs.py          # CIFAR-10 run summarizer
├── experiments/                     # Experiment command packs
├── report/                          # Report drafts and final report
├── datasets/                        # Local Dogs vs Cats and CIFAR-10 data
├── outputs/                         # Training outputs, checkpoints, submissions, and metrics
├── results/                         # Curated copies of final result summaries
├── environment.yml                  # CPU/default conda environment
├── environment.gpu.yml              # CUDA 12.8 GPU conda environment
├── requirements-cu128.txt           # PyTorch CUDA 12.8 pip packages
└── sampleSubmission.csv             # Sample submission format
```

For group sharing, this repository may include large artifacts such as `datasets/`, `outputs/`, and model checkpoints. This makes it easier for teammates and the marker to inspect the full project state. The curated final summaries are also copied into `results/` for quick review.

## Environment Setup

### GPU Setup

Recommended if you have an NVIDIA GPU with a recent driver:

```powershell
conda env create -f environment.gpu.yml
conda activate 6483_project
```

If you already have the environment and only need to install the CUDA PyTorch packages:

```powershell
conda activate 6483_project
python -m pip install -r requirements-cu128.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### CPU Setup

```powershell
conda env create -f environment.yml
conda activate 6483_project
```

CPU training works for debugging, but the full ResNet experiments are much faster on GPU.

## Data Layout

Dogs vs Cats data should be placed as:

```text
datasets/datasets/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
  test/
```

CIFAR-10 is downloaded automatically by `train_cifar10.py` when `--download` is used:

```text
datasets/cifar10/
```

The data folders are kept in the project so teammates can reproduce the experiments without re-downloading or reorganizing data.

## Dogs vs Cats

### Final Training Command

```powershell
python train.py `
  --data-dir datasets/datasets `
  --output-dir outputs/final_resnet34_full `
  --arch resnet34 `
  --weights imagenet `
  --epochs 8 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --num-workers 4 `
  --device cuda
```

### Generate Submission

```powershell
python predict.py `
  --checkpoint outputs/final_resnet34_full/best_model.pt `
  --test-dir datasets/datasets/test `
  --output-csv outputs/final_resnet34_full/submission.csv `
  --device cuda
```

### Dogs vs Cats Results

Final model:

- Model: ImageNet-pretrained ResNet34
- Training data: 20,000 images
- Validation data: 5,000 images
- Best validation accuracy: 99.02%
- Best epoch: 2
- Final submission: `results/dogs_vs_cats/submission.csv`

Model comparison summary:

| Model | Setting | Best validation accuracy |
|---|---|---:|
| SimpleCNN | trained from scratch | 63.50% |
| ResNet18 | pretrained, augmentation | 98.70% |
| ResNet18 | pretrained, no augmentation | 98.80% |
| ResNet34 | pretrained, augmentation | 99.20% |
| ResNet34 | full final training | 99.02% |

Validation error analysis:

- Correct predictions: 4,951 / 5,000
- Incorrect predictions: 49 / 5,000
- `cat -> dog` errors: 34
- `dog -> cat` errors: 15

See `report/error_sample_analysis.md` and `results/error_analysis/`.

## CIFAR-10

### Balanced CIFAR-10

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

Result:

- Model: ImageNet-pretrained ResNet18
- Test accuracy: 94.89%

### Class Imbalance Experiments

Imbalance setup:

- Majority classes `0-4`: 5,000 training images per class
- Minority classes `5-9`: 500 training images per class
- Test set remains balanced

| Method | Best test accuracy | Minority-class average |
|---|---:|---:|
| No correction | 90.40% | 84.42% |
| Class-weighted loss | 91.38% | 88.18% |
| Oversampling | 92.02% | 88.80% |

Run commands are in `experiments/cifar10_experiment_pack.md`.

## Report

The formal report draft is:

```text
report/final_report.md
```

Before submission, fill in the group member table:

- Name
- Email
- Matriculation No.

The report uses the following division of work:

- Member A: Data processing, Dogs vs Cats baseline, final submission generation.
- Member B: Literature survey, model comparison experiments, correct/incorrect sample analysis.
- Member C: CIFAR-10 extension, class imbalance experiments, report integration and formatting.

## Useful Commands

Summarize Dogs vs Cats runs:

```powershell
python summarize_runs.py `
  --outputs-dir outputs `
  --output-csv outputs/experiment_summary.csv
```

Summarize CIFAR-10 runs:

```powershell
python summarize_cifar_runs.py `
  --outputs-dir outputs `
  --output-csv outputs/cifar10_experiment_summary.csv
```

Run validation error analysis:

```powershell
python analyze_val_errors.py `
  --checkpoint outputs/final_resnet34_full/best_model.pt `
  --val-dir datasets/datasets/val `
  --output-dir outputs/final_resnet34_full/error_analysis `
  --device cuda
```

## GitHub Upload Notes

This project is intended to be shared with group members and referenced in the report. The repository keeps the full experiment state, including data and outputs. The `.gitignore` only excludes local/editor/cache files such as `.vscode/`, `__pycache__/`, and OS metadata files.

If GitHub is used through the command line:

```powershell
git init
git add .
git status
git commit -m "Add EE6483 project code and report"
```

Check `git status` before committing so you can confirm that the intended project files are included. Because this repository can be large, GitHub may require Git LFS for files larger than 100 MB, especially model checkpoint files in `outputs/`.
