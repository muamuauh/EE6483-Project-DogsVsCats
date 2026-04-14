# Dogs vs Cats Report Map

Use this file to connect your experiment outputs to the assignment questions.

## Data usage and preprocessing

Write down:

- exact training and validation counts used in the final run
- image size
- normalization
- augmentation choices

You can copy these from each run's `run_config.json`.

## Model description

For the baseline section, describe:

- architecture name, for example `ResNet18`
- whether pretrained ImageNet weights were used
- input size `224 x 224`
- output size `2`
- loss function: `CrossEntropyLoss`
- optimizer: `Adam`
- learning rate and weight decay

## Parameter discussion

Use your experiment comparisons to justify:

- why you kept or removed augmentation
- why you chose the final architecture
- why your learning rate and epoch count are reasonable

## Results table

Recommended table columns:

- run name
- model
- pretrained or not
- augmentation
- train size
- val size
- epochs
- best validation accuracy

You can generate most of this automatically with `summarize_runs.py`.

## Qualitative analysis

After you create the final `submission.csv`, choose a few validation images for:

- correct predictions
- incorrect predictions

Discuss:

- strong pose or clear foreground cases the model handles well
- cluttered background, unusual pose, occlusion, or low lighting cases it struggles with

## Suggested final baseline story

One clean report narrative is:

1. Start with pretrained `ResNet18` as the main baseline.
2. Compare against a custom CNN.
3. Test the effect of augmentation.
4. Optionally compare `ResNet18` with `ResNet34`.
5. Use the best-performing setup for the final submission model.
