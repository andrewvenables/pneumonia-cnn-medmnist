# Pneumonia detection from chest X-rays (PneumoniaMNIST)

A small convolutional neural network (PyTorch) that classifies chest X-rays
as normal or pneumonia, trained on the PneumoniaMNIST benchmark (28x28
grayscale pediatric X-rays). This is a learning project, not a diagnostic
tool. It was developed with AI assistance, and I ran, tested and analysed
it myself.

## Results

| Run | Accuracy | Pneumonia recall | Normal recall |
|---|---|---|---|
| Baseline | 0.82 | 0.99 | 0.53 |
| Class weights 1.67 | 0.82 | 1.00 | 0.51 |
| Class weights 2.88 | 0.83 | 0.99 | 0.56 |

Test set: 234 normal, 390 pneumonia. Training set: 1,214 normal, 3,494 pneumonia.

## What I found

- Accuracy alone is misleading here. The model catches nearly every
  pneumonia case (recall 0.99) but clears only about half of healthy
  X-rays (recall about 0.5), so it over-predicts pneumonia.
- Class weighting made little difference. The changes in normal recall
  (0.51 to 0.56) are small enough to be run-to-run noise.
- Validation accuracy (about 96%) was far above test accuracy (82%).
  My hypothesis is that the test images differ from the training data,
  so the model does not transfer well. I have not tested this.
- Training and test sets have different class balance (74% vs 62.5%
  pneumonia).

## Possible next steps

- Repeat each run with several random seeds to check whether the
  differences in normal recall are real or just noise.
- Test the validation-to-test gap hypothesis by comparing image
  statistics between the splits.
- Try data augmentation and a larger image size (64x64 or 128x128).

## How to run

1. Open [Google Colab](https://colab.research.google.com) and set the
   runtime to a GPU (Runtime > Change runtime type > T4 GPU).
2. In a cell, run: `!pip install medmnist`
3. Upload `pneumonia_classifier.py` (or paste its contents into a cell) and run it.
4. To see recall per class, run the classification report from
   scikit-learn on the test set after training.

Dependencies: torch, torchvision, medmnist, numpy, scikit-learn.
