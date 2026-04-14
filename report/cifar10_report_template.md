# CIFAR-10 Report Template

## Part (g): Multi-category CIFAR-10 classification

CIFAR-10 is a 10-class image classification dataset containing color images from the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Compared with the Dogs vs Cats task, CIFAR-10 is more challenging because it requires multi-class prediction instead of binary classification, and the original image size is only `32 x 32`.

We adapted the Dogs vs Cats classifier to CIFAR-10 by changing the final classification layer from 2 outputs to 10 outputs. The training pipeline still used a pretrained residual network as the feature extractor, followed by a task-specific classifier layer. The input images were resized to `224 x 224` to match the ImageNet-pretrained ResNet input convention. Data augmentation included random cropping and horizontal flipping during training, while the test set used deterministic resizing and normalization.

Recommended table:

| Run | Model | Training setting | Test accuracy |
|---|---|---|---:|
| `cifar10_resnet18_balanced` | ResNet18 | balanced training set | 94.89% |

## Part (h): CIFAR-10 with class imbalance

To study class imbalance, we constructed an imbalanced CIFAR-10 training set. Classes `0-4` were treated as majority classes, while classes `5-9` were treated as minority classes. The majority classes used `5000` labelled images per class, while the minority classes used only `500` labelled images per class. The official CIFAR-10 test set remained balanced, so the result reflects whether the model can generalize to all classes under imbalanced training.

We tested two imbalance-handling methods:

1. Class-weighted loss: higher loss weights were assigned to minority classes. This encourages the optimizer to pay more attention to mistakes on under-represented classes.
2. Oversampling: minority-class samples were sampled more frequently during training using a weighted sampler. This makes each mini-batch distribution more balanced without changing the original labels.

Recommended table:

| Run | Method | Majority / minority samples | Test accuracy | Notes |
|---|---|---:|---:|---|
| `cifar10_imb_none` | no correction | 5000 / 500 | 90.40% | baseline under imbalance |
| `cifar10_imb_weighted_loss` | class-weighted loss | 5000 / 500 | 91.38% | improved minority classes |
| `cifar10_imb_oversampling` | oversampling | 5000 / 500 | 92.02% | best overall imbalanced result |

Suggested discussion:

The imbalanced baseline is expected to perform worse on minority classes because the model sees far fewer examples from those classes during training. Class-weighted loss addresses this issue by increasing the penalty for minority-class errors, while oversampling addresses it by increasing the frequency of minority samples in training batches. These two methods improve the algorithm from different perspectives: one modifies the objective function, and the other modifies the data sampling distribution.

## Files to use

- `outputs/cifar10_experiment_summary.csv`
- `outputs/cifar10_resnet18_balanced/summary.json`
- `outputs/cifar10_imb_none/summary.json`
- `outputs/cifar10_imb_weighted_loss/summary.json`
- `outputs/cifar10_imb_oversampling/summary.json`
- `outputs/*/best_per_class_accuracy.json`
