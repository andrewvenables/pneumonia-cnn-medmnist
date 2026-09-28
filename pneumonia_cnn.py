"""
Pneumonia detection from chest X-rays (PneumoniaMNIST) - PyTorch CNN
====================================================================

Trains a small convolutional neural network to classify chest X-rays as
"normal" or "pneumonia" using the PneumoniaMNIST benchmark (28x28 grayscale
pediatric X-rays, from the MedMNIST collection).

This is a learning project, not a diagnostic tool. It was developed with AI
assistance; I ran, tested and analysed it myself (see README for results).

Run in Google Colab (GPU optional):
    !pip install medmnist
Then run this file, or paste it into a cell.

To reproduce the baseline run, set USE_CLASS_WEIGHTS = False.
"""

# -- 1. SETUP ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from sklearn.metrics import classification_report
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

USE_CLASS_WEIGHTS = True   # False = baseline run
N_EPOCHS = 15


# -- 2. LOAD THE DATA -------------------------------------------------------
# PneumoniaMNIST: pediatric chest X-rays labelled normal (0) or pneumonia (1).
train_dataset = PneumoniaMNIST(split="train", download=True, size=28)
val_dataset   = PneumoniaMNIST(split="val",   download=True, size=28)
test_dataset  = PneumoniaMNIST(split="test",  download=True, size=28)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


def to_tensor_dataset(dataset):
    """Convert MedMNIST PIL images to normalised tensors in the range [-1, 1]."""
    images, labels = [], []
    for img, label in dataset:
        arr = np.array(img, dtype=np.float32) / 255.0   # 0-255 -> 0-1
        arr = (arr - 0.5) / 0.5                          # 0-1 -> -1 to 1
        images.append(arr)
        labels.append(label[0])

    images = torch.tensor(np.array(images)).unsqueeze(1)  # add channel dim: [N,1,28,28]
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    return torch.utils.data.TensorDataset(images, labels)


train_data = to_tensor_dataset(train_dataset)
val_data   = to_tensor_dataset(val_dataset)
test_data  = to_tensor_dataset(test_dataset)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)


# -- 3. DEFINE THE MODEL ----------------------------------------------------
# Small filters slide over the image to detect local patterns (edges,
# textures) in early layers; later layers combine them into larger patterns.
class PneumoniaCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: simple local patterns
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),   # stabilises training
            nn.ReLU(),            # non-linearity
            nn.MaxPool2d(2),      # 28x28 -> 14x14

            # Block 2: more complex patterns
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),      # randomly zeroes 30% of activations to reduce overfitting
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = PneumoniaCNN().to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} trainable parameters")


# -- 4. TRAINING SETUP ------------------------------------------------------
# Class balance in the TRAINING set (weights come from here, not the test set)
train_counts = np.bincount([int(l) for _, l in train_data])
print(f"Train, normal: {train_counts[0]} | pneumonia: {train_counts[1]}")

if USE_CLASS_WEIGHTS:
    # Give the smaller "normal" class a proportionally bigger weight
    weights = torch.tensor([train_counts[1] / train_counts[0], 1.0],
                           dtype=torch.float32).to(device)
    print(f"Class weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()      # compute gradients
                optimizer.step()     # update weights

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -- 5. TRAIN THE MODEL -----------------------------------------------------
best_val_acc = 0

for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch:2d}/{N_EPOCHS} | "
          f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.3f} acc {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")


# -- 6. FINAL EVALUATION ON THE HELD-OUT TEST SET ---------------------------
model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"\nFinal test accuracy: {test_acc:.3f}")

# Accuracy alone is not enough for a medical task: a missed pneumonia case
# (false negative) is worse than a false alarm, so look at recall per class.
counts = np.bincount([int(l) for _, l in test_data])
print(f"Normal: {counts[0]} | Pneumonia: {counts[1]}")

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print(classification_report(all_labels, all_preds, target_names=["normal", "pneumonia"]))
