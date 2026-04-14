# Final Results Summary

This file summarizes the final `Dogs vs Cats` baseline results using the actual experiment outputs in this repository.

## Final model

- Architecture: `ResNet34`
- Initialization: pretrained `ImageNet` weights
- Input size: `224 x 224`
- Batch size: `32`
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Epochs trained: `8`
- Device: `CUDA`
- Data augmentation: enabled
- Training set size: `20,000` images
- Validation set size: `5,000` images
- Label mapping: `cat = 0`, `dog = 1`

## Final performance

- Best validation accuracy: `0.9902` (`99.02%`)
- Best epoch: `epoch 2`
- Total training time: `10 minutes 05 seconds`
- Best checkpoint: `outputs/final_resnet34_full/best_model.pt`
- Submission file: `outputs/final_resnet34_full/submission.csv`

## Training dynamics

From `history.csv`:

- Epoch 1: validation accuracy `0.9884`
- Epoch 2: validation accuracy `0.9902`
- Epoch 3: validation accuracy `0.9824`
- Epoch 4: validation accuracy `0.9860`
- Epoch 5: validation accuracy `0.9854`
- Epoch 6: validation accuracy `0.9858`
- Epoch 7: validation accuracy `0.9880`
- Epoch 8: validation accuracy `0.9842`

Interpretation:

- The model converged quickly and already achieved very strong performance by epoch 2.
- After epoch 2, training accuracy continued to improve while validation accuracy slightly fluctuated, which suggests mild overfitting.
- Because the training script saves the best checkpoint automatically, the best-performing model was preserved.

## Experiment comparison

Main completed runs:

| Run | Model | Pretrained | Augmentation | Train / Val size | Best val accuracy |
|---|---|---|---|---|---|
| `exp_a_resnet18_baseline` | ResNet18 | Yes | Yes | 4000 / 1000 | `0.9870` |
| `exp_b_resnet18_no_aug` | ResNet18 | Yes | No | 4000 / 1000 | `0.9880` |
| `exp_c_simple_cnn` | SimpleCNN | No | Yes | 4000 / 1000 | `0.6350` |
| `exp_d_resnet34` | ResNet34 | Yes | Yes | 4000 / 1000 | `0.9920` |
| `final_resnet34_full` | ResNet34 | Yes | Yes | 20000 / 5000 | `0.9902` |

## Recommended conclusions for the report

1. A pretrained residual network is much more effective than a small CNN for this task.
2. `SimpleCNN` only achieved around `63.5%` validation accuracy, showing that a shallow model trained from scratch was not sufficient.
3. `ResNet18` and `ResNet34` both performed very strongly, and both exceeded `98%` validation accuracy.
4. Among the tested models, `ResNet34` achieved the best validation performance and was therefore chosen as the final model.
5. In the `ResNet18` comparison, disabling augmentation slightly improved validation accuracy (`98.8%` vs `98.7%`). This suggests the current augmentation strategy did not provide a measurable gain under that setting.
6. On the full dataset, the final `ResNet34` model achieved `99.02%` validation accuracy, which confirms strong generalization on the Dogs vs Cats task.

## Suggested paragraph for the report

We compared several image classification baselines for the Dogs vs Cats task, including a custom CNN, a pretrained ResNet18, and a pretrained ResNet34. The custom CNN only achieved `63.5%` validation accuracy, while the pretrained residual networks exceeded `98%`, indicating that transfer learning provided a significant performance benefit. Among all tested settings, ResNet34 achieved the best result and was therefore selected as the final model. Using the full training and validation sets, the final ResNet34 model reached a best validation accuracy of `99.02%` at epoch 2. After that point, the training accuracy continued to increase while validation accuracy fluctuated slightly, suggesting mild overfitting. Therefore, the checkpoint with the best validation performance was used to generate the final `submission.csv`.

## Files used for this summary

- `outputs/final_resnet34_full/run_config.json`
- `outputs/final_resnet34_full/summary.json`
- `outputs/final_resnet34_full/history.csv`
- `outputs/experiment_summary.csv`
