
# EE6483 Artificial Intelligence and Data Mining

# Mini Project Option 2: Dogs vs Cats Image Classification

## Cover Page

**Course:** EE6483 Artificial Intelligence and Data Mining  
**Project:** Mini Project Option 2: Dogs vs Cats  
**Task:** Binary Dogs vs Cats Classification, CIFAR-10 Extension, and CIFAR-10 Class Imbalance Study  
**GitHub Repository:** https://github.com/muamuauh/EE6483-Project-DogsVsCats  
**Submission File:** `outputs/final_resnet34_full/submission.csv`

## Table of Contents

1. [Group Information](#group-information)
2. [Abstract](#abstract)
3. [Literature Survey](#1-literature-survey)
4. [Dataset Usage and Preprocessing](#2-dataset-usage-and-preprocessing)
5. [Model and Training Strategy](#3-model-and-training-strategy)
6. [Parameter Selection](#4-parameter-selection)
7. [Dogs vs Cats Results](#5-dogs-vs-cats-results)
8. [Correct and Incorrect Sample Analysis](#6-correct-and-incorrect-sample-analysis)
9. [Effect of Model Choice and Data Processing](#7-effect-of-model-choice-and-data-processing)
10. [CIFAR-10 Extension](#8-cifar-10-extension)
11. [CIFAR-10 Class Imbalance Experiment](#9-cifar-10-class-imbalance-experiment)
12. [Conclusion](#10-conclusion)
13. [References](#references)

## Group Information

| Role | Name | Email | Matriculation No. | Main contribution |
|---|---|---|---|---|
| Member A | [Name] | [Email] | [Matriculation No.] | Data processing, Dogs vs Cats baseline, final submission generation |
| Member B | [Name] | [Email] | [Matriculation No.] | Literature survey, model comparison experiments, correct/incorrect sample analysis |
| Member C | [Name] | [Email] | [Matriculation No.] | CIFAR-10 extension, class imbalance experiments, report integration and formatting |

## Abstract

This project studies image classification for the Dogs vs Cats dataset and extends the same classification framework to CIFAR-10. For the binary Dogs vs Cats task, we compared a simple CNN, pretrained ResNet18, and pretrained ResNet34 models. The final selected model was a pretrained ResNet34 trained on the full Dogs vs Cats training set of 20,000 images and evaluated on 5,000 validation images. It achieved 99.02% validation accuracy, and the best checkpoint was used to generate `submission.csv` for the 500-image test set. For the CIFAR-10 extension, we adapted the classifier to 10 classes using pretrained ResNet18 and achieved 94.89% test accuracy. We further studied class imbalance by reducing classes 5-9 to 500 training samples per class while keeping classes 0-4 at 5,000 samples per class. Under imbalance, no correction achieved 90.40%, class-weighted loss achieved 91.38%, and oversampling achieved 92.02% test accuracy.

## 1. Literature Survey

### 1.1 Problem Definition

Image classification assigns an input image to one of a set of semantic categories. In this project, the first problem is a supervised closed-set binary classification task: each training image is labelled as cat or dog, and all validation/test images are assumed to belong to one of these two classes. The second problem extends the same idea to the supervised closed-set multi-class CIFAR-10 task, which has 10 mutually exclusive categories.

Different settings lead to different solution strategies:

- Supervised classification: labelled images are available, so models are trained by minimizing a classification loss such as cross entropy.
- Unsupervised or self-supervised learning: labels are unavailable or limited, so representation learning methods are used before downstream classification.
- Closed-set recognition: test classes are known at training time, as in Dogs vs Cats and CIFAR-10.
- Open-set recognition: test images may contain unknown classes, requiring confidence estimation or rejection mechanisms.
- Without domain shift: train and test data share similar distributions, so standard supervised learning is usually effective.
- With domain shift: train and test images may differ in background, style, lighting, resolution, or source domain, requiring augmentation, domain adaptation, or robust representations.

### 1.2 Development of Image Classification Methods

Early modern deep-learning image classification was strongly influenced by AlexNet, which showed that deep CNNs trained with GPU acceleration could substantially improve ImageNet classification results [1]. VGG then showed that increasing network depth with small 3x3 filters could improve representation quality and transferability [2]. ResNet introduced residual connections, making deeper neural networks easier to optimize and achieving strong results on ImageNet and CIFAR-10 [3]. These developments are directly relevant to this project because both Dogs vs Cats and CIFAR-10 are natural-image classification tasks where pretrained convolutional features transfer well.

More recent work has explored model scaling and transformer-based image recognition. EfficientNet proposed compound scaling of depth, width, and resolution, showing that carefully balanced scaling improves accuracy and efficiency [4]. Vision Transformer (ViT) showed that pure transformer architectures can perform very well for image classification when pretrained at sufficient scale [5]. ConvNeXt revisited convolutional architectures under modern design and training practices, showing that pure ConvNets can remain competitive with transformer-style models [6]. These papers indicate two major trends: stronger pretrained backbones and better scaling/training strategies.

### 1.3 Suitable Baseline for This Project

For this project, a pretrained ResNet is a suitable baseline because it balances accuracy, training speed, implementation simplicity, and report interpretability. A custom CNN is easy to understand but performed poorly in our experiments. Larger modern models such as ViT or EfficientNet may achieve strong results but add more complexity. Therefore, we used pretrained ResNet18/ResNet34 models as the main approach and used a custom CNN as a comparison baseline.

## 2. Dataset Usage and Preprocessing

### 2.1 Dogs vs Cats Dataset

The dataset follows the required directory structure:

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

The full available data were:

| Split | Cat images | Dog images | Total |
|---|---:|---:|---:|
| Train | 10,000 | 10,000 | 20,000 |
| Validation | 2,500 | 2,500 | 5,000 |
| Test | unlabeled | unlabeled | 500 |

The final Dogs vs Cats model used all 20,000 training images and all 5,000 validation images. The 500 test images were used only for prediction after training.

### 2.2 Preprocessing and Augmentation

For Dogs vs Cats, the final preprocessing pipeline was:

- Resize images to 224x224.
- Convert images to tensor format.
- Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- During training, use random horizontal flip and random rotation.
- During validation and testing, use deterministic resize and normalization only.

For CIFAR-10, the original 32x32 images were resized to 224x224 to match the ImageNet-pretrained ResNet input convention. Training augmentation included random crop and horizontal flip, followed by the same ImageNet normalization.

## 3. Model and Training Strategy

### 3.1 Model Architecture

The final Dogs vs Cats model was a pretrained ResNet34 with its final fully connected layer replaced by a 2-class output layer.

```text
Input image: 3 x 224 x 224
        |
ImageNet-pretrained ResNet34 convolutional backbone
        |
Global average pooling
        |
Fully connected layer: 512 -> 2
        |
Output logits: [cat, dog]
```

For CIFAR-10, the same idea was used with pretrained ResNet18, but the final layer was changed to 10 outputs:

```text
Input image: 3 x 224 x 224
        |
ImageNet-pretrained ResNet18 convolutional backbone
        |
Global average pooling
        |
Fully connected layer: 512 -> 10
        |
Output logits: CIFAR-10 classes
```

### 3.2 Loss Function and Optimizer

The training loss was `CrossEntropyLoss`, which is appropriate for both binary and multi-class classification when the model outputs class logits. The optimizer was Adam with learning rate `1e-4` and weight decay `1e-4`. A fixed random seed of `42` was used for reproducibility.

### 3.3 Code and Running Instructions

Main code files:

- `train.py`: Dogs vs Cats training.
- `predict.py`: Dogs vs Cats test prediction and `submission.csv` generation.
- `baseline_utils.py`: shared dataset, transform, and model utilities.
- `analyze_val_errors.py`: validation correct/incorrect sample analysis.
- `train_cifar10.py`: CIFAR-10 balanced and imbalanced training.
- `summarize_runs.py`: Dogs vs Cats experiment summary.
- `summarize_cifar_runs.py`: CIFAR-10 experiment summary.

Environment:

```powershell
conda activate 6483_project
cd project folder
```

Final Dogs vs Cats training command:

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

Final Dogs vs Cats prediction command:

```powershell
python predict.py `
  --checkpoint outputs/final_resnet34_full/best_model.pt `
  --test-dir datasets/datasets/test `
  --output-csv outputs/final_resnet34_full/submission.csv `
  --device cuda
```

CIFAR-10 balanced training command:

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
  --device cuda
```

## 4. Parameter Selection

We selected ImageNet-pretrained ResNet models because transfer learning is effective for natural-image recognition and reduces the amount of training needed. The input size was set to 224x224 because it matches the common ImageNet-pretrained ResNet setting. The learning rate was set to `1e-4` to provide stable fine-tuning. The batch size was set to 32 for Dogs vs Cats and 64 for CIFAR-10 after GPU testing. Weight decay `1e-4` was used as regularization. For Dogs vs Cats, 8 epochs were used for final full-data training, but the best checkpoint was selected automatically based on validation accuracy.

## 5. Dogs vs Cats Results

### 5.1 Model Comparison

Initial controlled experiments used 4,000 training images and 1,000 validation images.

| Run | Model | Pretrained | Augmentation | Train / Val size | Best validation accuracy |
|---|---|---|---|---:|---:|
| `exp_a_resnet18_baseline` | ResNet18 | Yes | Yes | 4000 / 1000 | 98.70% |
| `exp_b_resnet18_no_aug` | ResNet18 | Yes | No | 4000 / 1000 | 98.80% |
| `exp_c_simple_cnn` | SimpleCNN | No | Yes | 4000 / 1000 | 63.50% |
| `exp_d_resnet34` | ResNet34 | Yes | Yes | 4000 / 1000 | 99.20% |

The custom CNN performed much worse than pretrained ResNet models, confirming that pretrained deep features were important for this task. ResNet34 achieved the highest validation accuracy among the comparison models and was selected for final training.

### 5.2 Final Dogs vs Cats Result

The final ResNet34 model was trained on all 20,000 training images and evaluated on all 5,000 validation images.

| Model | Training images | Validation images | Best validation accuracy | Best epoch | Training time |
|---|---:|---:|---:|---:|---:|
| ResNet34 pretrained on ImageNet | 20,000 | 5,000 | 99.02% | 2 | 10:05 |

The model converged quickly. Validation accuracy reached 98.84% after epoch 1 and peaked at 99.02% at epoch 2. Later epochs showed slightly lower validation accuracy while training accuracy continued to increase, suggesting mild overfitting. The best checkpoint at epoch 2 was used for final prediction.

### 5.3 Submission File

The final submission file is:

```text
outputs/final_resnet34_full/submission.csv
```

It contains two columns:

```csv
id,label
```

The label mapping is:

- `0 = cat`
- `1 = dog`

## 6. Correct and Incorrect Sample Analysis

The final ResNet34 model achieved 99.02% validation accuracy on 5,000 validation images:

- Correct predictions: 4,951
- Incorrect predictions: 49

Confusion matrix:

| True label | Predicted cat | Predicted dog | Class accuracy |
|---|---:|---:|---:|
| cat | 2466 | 34 | 98.64% |
| dog | 15 | 2485 | 99.40% |

Representative high-confidence correct predictions included `cat.10280.jpg`, `cat.10525.jpg`, `cat.1064.jpg`, `dog.10029.jpg`, `dog.10042.jpg`, and `dog.10124.jpg`, all predicted with confidence 1.0000. These examples show that the final model can strongly recognize clear class-discriminative cat and dog images.

Representative high-confidence incorrect predictions included:

| Image | True label | Predicted label | Confidence |
|---|---|---|---:|
| `cat.11565.jpg` | cat | dog | 1.0000 |
| `cat.7920.jpg` | cat | dog | 0.9996 |
| `cat.11149.jpg` | cat | dog | 0.9973 |
| `dog.9109.jpg` | dog | cat | 0.9853 |
| `dog.1568.jpg` | dog | cat | 0.9718 |

These high-confidence mistakes indicate that the model is not merely uncertain in every failure case. Some images contain misleading visual cues that strongly activate the wrong class. Possible causes include unusual pose, partial occlusion, background bias, image crop, or texture similarity. Low-confidence correct predictions such as `cat.2159.jpg` and `dog.6199.jpg` also show that some samples lie close to the decision boundary.

Generated analysis files:

- `outputs/final_resnet34_full/error_analysis/val_predictions.csv`
- `outputs/final_resnet34_full/error_analysis/confusion_matrix.csv`
- `outputs/final_resnet34_full/error_analysis/incorrect_contact_sheet.jpg`
- `outputs/final_resnet34_full/error_analysis/correct_contact_sheet.jpg`

## 7. Effect of Model Choice and Data Processing

Model choice had the largest impact. The simple CNN achieved only 63.50% validation accuracy, while pretrained ResNet18 and ResNet34 exceeded 98%. This shows that transfer learning from ImageNet was much more effective than training a small CNN from scratch on this task.

Data augmentation had a smaller and less consistent effect in our ResNet18 comparison. With augmentation, ResNet18 achieved 98.70%; without augmentation, it achieved 98.80%. This difference is small, suggesting that the pretrained ResNet already generalized well under this data split and that the chosen augmentation strategy was not a major performance driver. However, augmentation was retained in the final ResNet34 model as a conservative regularization choice.

Increasing model capacity from ResNet18 to ResNet34 improved the subset experiment from 98.70% to 99.20%, so ResNet34 was selected for final training. On the full dataset, the final ResNet34 model achieved 99.02% validation accuracy.

## 8. CIFAR-10 Extension

CIFAR-10 contains 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images. The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck [7]. We adapted the Dogs vs Cats classifier by changing the output dimension from 2 to 10. The model used pretrained ResNet18 with ImageNet normalization and 224x224 resizing.

Balanced CIFAR-10 result:

| Run | Model | Training set | Test set | Best test accuracy | Best epoch |
|---|---|---:|---:|---:|---:|
| `cifar10_resnet18_balanced` | ResNet18 pretrained on ImageNet | 50,000 | 10,000 | 94.89% | 10 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| airplane | 97.20% |
| automobile | 97.20% |
| bird | 92.30% |
| cat | 82.20% |
| deer | 93.90% |
| dog | 94.70% |
| frog | 98.40% |
| horse | 97.30% |
| ship | 97.90% |
| truck | 97.80% |

The results show that the same transfer-learning framework can be extended from binary classification to multi-class classification by modifying the output layer and using the appropriate dataset transforms.

## 9. CIFAR-10 Class Imbalance Experiment

To study class imbalance, we constructed an imbalanced training set:

- Majority classes: airplane, automobile, bird, cat, deer, each with 5,000 training images.
- Minority classes: dog, frog, horse, ship, truck, each with 500 training images.
- Test set: unchanged balanced CIFAR-10 test set of 10,000 images.

We compared no correction, class-weighted loss, and oversampling.

| Run | Method | Train size | Majority / minority per class | Best test accuracy | Best epoch |
|---|---|---:|---:|---:|---:|
| `cifar10_imb_none` | no correction | 27,500 | 5000 / 500 | 90.40% | 9 |
| `cifar10_imb_weighted_loss` | class-weighted loss | 27,500 | 5000 / 500 | 91.38% | 7 |
| `cifar10_imb_oversampling` | oversampling | 27,500 | 5000 / 500 | 92.02% | 9 |

Per-class accuracy under imbalance:

| Class | No correction | Weighted loss | Oversampling |
|---|---:|---:|---:|
| airplane | 97.10% | 97.60% | 97.50% |
| automobile | 97.40% | 97.90% | 96.90% |
| bird | 94.90% | 93.90% | 92.80% |
| cat | 96.40% | 88.10% | 92.90% |
| deer | 96.10% | 95.40% | 96.10% |
| dog | 67.80% | 79.20% | 75.10% |
| frog | 84.40% | 91.30% | 95.30% |
| horse | 85.80% | 90.50% | 88.00% |
| ship | 90.50% | 89.70% | 92.30% |
| truck | 93.60% | 90.20% | 93.30% |

Average group accuracy:

| Method | Majority-class average | Minority-class average |
|---|---:|---:|
| No correction | 96.38% | 84.42% |
| Weighted loss | 94.58% | 88.18% |
| Oversampling | 95.24% | 88.80% |

Class imbalance clearly reduced performance: balanced CIFAR-10 achieved 94.89%, while the imbalanced baseline without correction dropped to 90.40%. Both imbalance-handling methods improved performance. Class-weighted loss increased minority-class average accuracy from 84.42% to 88.18%. Oversampling achieved the best overall result, with 92.02% test accuracy and 88.80% minority-class average accuracy.

Class-weighted loss changes the objective function by giving higher penalty to minority-class errors. Oversampling changes the sampling distribution by presenting minority-class examples more often during training. These two methods address imbalance from different perspectives, and both were useful in this experiment.

## 10. Conclusion

For Dogs vs Cats, pretrained ResNet models were much stronger than a small CNN trained from scratch. The final selected pretrained ResNet34 model achieved 99.02% validation accuracy on the full validation set and was used to generate the final `submission.csv`. Error analysis showed that only 49 out of 5,000 validation images were misclassified, with more cat-to-dog errors than dog-to-cat errors.

For CIFAR-10, the same transfer-learning framework was successfully extended to a 10-class classification problem, achieving 94.89% test accuracy with pretrained ResNet18. Under artificial class imbalance, performance dropped to 90.40% without correction. Weighted loss and oversampling improved results to 91.38% and 92.02%, respectively, showing that imbalance-aware training improves minority-class performance.

## References

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS, 2012. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

[2] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014. https://arxiv.org/abs/1409.1556

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019. https://research.google/pubs/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks/

[5] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021. https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/

[6] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," CVPR, 2022. https://arxiv.org/abs/2201.03545

[7] A. Krizhevsky, V. Nair, and G. Hinton, "CIFAR-10 and CIFAR-100 datasets." https://www.cs.toronto.edu/~kriz/cifar.html

[8] Kaggle, "Dogs vs. Cats." https://www.kaggle.com/c/dogs-vs-cats
