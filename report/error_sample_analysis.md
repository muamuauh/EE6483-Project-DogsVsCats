# Error Sample Analysis

This section is based on the final model `outputs/final_resnet34_full/best_model.pt`.

## Overall validation result

The final ResNet34 model achieved `99.02%` validation accuracy on `5,000` validation images:

- Correct predictions: `4,951`
- Incorrect predictions: `49`
- Total validation samples: `5,000`

The confusion matrix is:

| True label | Predicted cat | Predicted dog | Class accuracy |
|---|---:|---:|---:|
| cat | 2466 | 34 | 98.64% |
| dog | 15 | 2485 | 99.40% |

The model made more `cat -> dog` mistakes than `dog -> cat` mistakes. This suggests that, for the remaining difficult samples, some cat images were more likely to contain dog-like visual cues or ambiguous features that pushed the model toward the dog class.

## Correct sample analysis

Representative high-confidence correct predictions:

| Image | True label | Predicted label | Confidence |
|---|---|---|---:|
| `cat.10280.jpg` | cat | cat | 1.0000 |
| `cat.10525.jpg` | cat | cat | 1.0000 |
| `cat.1064.jpg` | cat | cat | 1.0000 |
| `dog.10029.jpg` | dog | dog | 1.0000 |
| `dog.10042.jpg` | dog | dog | 1.0000 |
| `dog.10124.jpg` | dog | dog | 1.0000 |

These examples show the strength of the final ResNet34 classifier: when the image contains class-discriminative features that align well with the learned representation, the model gives highly confident predictions for both cats and dogs. This supports the choice of a pretrained residual network as the final model, because it can extract robust visual features from natural images.

## Incorrect sample analysis

Representative high-confidence incorrect predictions:

| Image | True label | Predicted label | Confidence |
|---|---|---|---:|
| `cat.11565.jpg` | cat | dog | 1.0000 |
| `cat.7920.jpg` | cat | dog | 0.9996 |
| `cat.11149.jpg` | cat | dog | 0.9973 |
| `cat.6655.jpg` | cat | dog | 0.9779 |
| `cat.9882.jpg` | cat | dog | 0.9727 |
| `dog.9109.jpg` | dog | cat | 0.9853 |
| `dog.1568.jpg` | dog | cat | 0.9718 |
| `dog.9713.jpg` | dog | cat | 0.8936 |
| `dog.6489.jpg` | dog | cat | 0.8805 |
| `dog.10989.jpg` | dog | cat | 0.8646 |

The high-confidence wrong predictions are especially informative. They indicate that the model is not merely uncertain in all failure cases; in some samples, the visual features strongly activate the wrong class. This is a typical limitation of image classifiers trained only with image-level labels: the model may focus on correlated features such as pose, texture, background, cropping, or partial animal appearance rather than a complete semantic understanding of the object.

## Low-confidence correct cases

Some predictions were correct but close to the decision boundary:

| Image | True label | Predicted label | Confidence |
|---|---|---|---:|
| `cat.2159.jpg` | cat | cat | 0.5000 |
| `dog.6199.jpg` | dog | dog | 0.5040 |
| `cat.8198.jpg` | cat | cat | 0.5223 |
| `dog.12353.jpg` | dog | dog | 0.5287 |
| `cat.7122.jpg` | cat | cat | 0.5312 |
| `cat.10441.jpg` | cat | cat | 0.5665 |

These examples show that even when the final prediction is correct, the classifier can be uncertain. Such samples are useful for understanding the decision boundary: they may contain less typical views, unusual backgrounds, partial occlusions, or visual ambiguity between cats and dogs.

## Report-ready paragraph

To further analyze the model behavior, we examined both correctly and incorrectly classified validation samples. The final ResNet34 model correctly classified `4,951` out of `5,000` validation images, achieving `99.02%` validation accuracy. The confusion matrix shows that `34` cat images were misclassified as dogs, while `15` dog images were misclassified as cats, indicating that the model made slightly more mistakes on cat images. High-confidence correct cases such as `cat.10280.jpg` and `dog.10029.jpg` demonstrate that the model can reliably classify clear class-discriminative images. However, high-confidence failure cases such as `cat.11565.jpg` and `dog.9109.jpg` show that the classifier can still make confident mistakes when the image contains misleading visual cues. In addition, low-confidence correct predictions such as `cat.2159.jpg` and `dog.6199.jpg` suggest that some samples lie close to the decision boundary. Overall, the model is highly accurate, but its remaining weaknesses are likely associated with visually ambiguous, atypical, or misleading samples.

## Generated files

- `outputs/final_resnet34_full/error_analysis/val_predictions.csv`
- `outputs/final_resnet34_full/error_analysis/confusion_matrix.csv`
- `outputs/final_resnet34_full/error_analysis/selected_samples.md`
- `outputs/final_resnet34_full/error_analysis/balanced_selected_samples.md`
- `outputs/final_resnet34_full/error_analysis/incorrect_contact_sheet.jpg`
- `outputs/final_resnet34_full/error_analysis/correct_contact_sheet.jpg`
- `outputs/final_resnet34_full/error_analysis/low_confidence_correct_contact_sheet.jpg`
