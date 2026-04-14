# CIFAR-10 Final Results

This section covers assignment parts (g) and (h).

## Part (g): Multi-category CIFAR-10 classification

CIFAR-10 is a 10-category image classification dataset with the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Compared with the Dogs vs Cats task, CIFAR-10 changes the problem from binary classification to multi-class classification. Therefore, the classifier output layer was changed from `2` outputs to `10` outputs.

We reused the transfer-learning strategy from Dogs vs Cats. A pretrained `ResNet18` backbone was used, and the final fully connected layer was replaced with a 10-class classifier. CIFAR-10 images were resized from `32 x 32` to `224 x 224` to match the ImageNet-pretrained ResNet input convention. During training, random crop and horizontal flip were used as data augmentation. The model was trained using `CrossEntropyLoss` and the Adam optimizer with learning rate `1e-4`.

Final balanced CIFAR-10 result:

| Run | Model | Training set | Test set | Best test accuracy | Best epoch |
|---|---|---:|---:|---:|---:|
| `cifar10_resnet18_balanced` | ResNet18 pretrained on ImageNet | 50,000 | 10,000 | 94.89% | 10 |

Per-class accuracy for the balanced run:

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

The results show that the pretrained ResNet18 can be successfully adapted from binary Dogs vs Cats classification to 10-class CIFAR-10 classification. Most classes achieved high test accuracy, while the cat class was more challenging than others in this run.

## Part (h): CIFAR-10 with class imbalance

To evaluate class imbalance, we constructed an imbalanced CIFAR-10 training set. Classes `0-4` were treated as majority classes and kept at `5000` samples per class. Classes `5-9` were treated as minority classes and reduced to `500` samples per class. The official test set remained balanced with `10000` images.

The class split was:

- Majority classes: airplane, automobile, bird, cat, deer
- Minority classes: dog, frog, horse, ship, truck

We compared three training settings:

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

Average accuracy by group:

| Method | Majority-class average | Minority-class average |
|---|---:|---:|
| No correction | 96.38% | 84.42% |
| Weighted loss | 94.58% | 88.18% |
| Oversampling | 95.24% | 88.80% |

## Discussion

Class imbalance clearly reduced performance. The balanced CIFAR-10 model achieved `94.89%` test accuracy, while the imbalanced model without correction dropped to `90.40%`. The minority-class average accuracy was only `84.42%` without correction, showing that the model was biased toward majority classes.

Both imbalance-handling approaches improved the imbalanced baseline. Class-weighted loss improved overall accuracy from `90.40%` to `91.38%` and increased the minority-class average from `84.42%` to `88.18%`. Oversampling achieved the best overall result, improving test accuracy to `92.02%` and minority-class average accuracy to `88.80%`.

The two approaches address imbalance in different ways. Class-weighted loss modifies the objective function so that mistakes on minority classes receive larger penalties. Oversampling modifies the training data distribution so that minority samples appear more frequently in mini-batches. In our experiments, oversampling performed slightly better overall, while both methods improved minority-class performance compared with no correction.

## Report-ready paragraph

For the CIFAR-10 extension, we adapted the Dogs vs Cats classifier to a 10-class classification problem by replacing the final classifier layer with a 10-output layer. Using a pretrained ResNet18 and ImageNet-style preprocessing, the balanced CIFAR-10 experiment achieved `94.89%` test accuracy. To study class imbalance, we reduced classes `5-9` to `500` training images per class while keeping classes `0-4` at `5000` training images per class. Without any correction, the model achieved `90.40%` test accuracy and the minority-class average accuracy was only `84.42%`. Applying class-weighted loss improved the test accuracy to `91.38%`, while oversampling further improved it to `92.02%`. These results show that class imbalance harms minority-class performance, and both weighted loss and oversampling can partially address the issue.

## Files

- `outputs/cifar10_experiment_summary.csv`
- `outputs/cifar10_resnet18_balanced/summary.json`
- `outputs/cifar10_imb_none/summary.json`
- `outputs/cifar10_imb_weighted_loss/summary.json`
- `outputs/cifar10_imb_oversampling/summary.json`
- `outputs/cifar10_resnet18_balanced/best_per_class_accuracy.json`
- `outputs/cifar10_imb_none/best_per_class_accuracy.json`
- `outputs/cifar10_imb_weighted_loss/best_per_class_accuracy.json`
- `outputs/cifar10_imb_oversampling/best_per_class_accuracy.json`
