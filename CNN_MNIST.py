# train_cnn_mnist.py
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# -----------------------
# 1) Config
# -----------------------
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
DATA_ROOT = "./data"
CKPT_DIR = Path("./checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# -----------------------
# 2) Dataset & Dataloader
# -----------------------
# ToTensor: [0,255] -> [0,1]; Normalize 用的是 MNIST 的常用均值/方差
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")


# -----------------------
# 3) Model (A simple CNN)
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: [B, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> [B, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> [B, 64, 28, 28]
        self.pool  = nn.MaxPool2d(2, 2)                           # -> [B, 64, 14, 14]
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# -----------------------
# 4) Train & Evaluate
# -----------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


# -----------------------
# 5) Run
# -----------------------
best_acc = 0.0
start = time.time()
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(epoch)
    test_loss, test_acc = evaluate()

    print(f"[EPOCH {epoch+1}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")

    # Save best
    if test_acc > best_acc:
        best_acc = test_acc
        ckpt_path = CKPT_DIR / "mnist_cnn_best.pt"
        torch.save({
            "model_state": model.state_dict(),
            "acc": best_acc,
            "epoch": epoch + 1
        }, ckpt_path)
        print(f"[INFO] Saved best checkpoint to {ckpt_path} (acc={best_acc:.2f}%)")

elapsed = time.time() - start
print(f"[DONE] Best test acc: {best_acc:.2f}% | elapsed: {elapsed:.1f}s")
