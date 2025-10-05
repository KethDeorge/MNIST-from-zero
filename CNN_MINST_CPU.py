# -*- coding: utf-8 -*-
"""
MNIST CNN（PyTorch）
-------------------------------------------------
功能：
1) 下载并加载完整 MNIST 数据集
2) 构建轻量 CNN 模型（两层卷积 + 池化 + 两层全连接）
3) 训练若干轮，评估测试集精度
4) 保存测试精度最好的模型权重到 ./checkpoints/mnist_cnn_best.pt
"""

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ========= 1) 基础配置 =========
SEED = 42  # 固定随机种子
DATA_ROOT = "./data"
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

# 线程（根据机器配置适当调整）
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
TORCH_NUM_INTEROP = int(os.environ.get("TORCH_NUM_INTEROP", "4"))

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= 2) 数据集 & DataLoader =========
transform = transforms.ToTensor()

# 使用完整 MNIST 数据集
train_ds = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

NUM_WORKERS = 4
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
test_loader  = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

# ========= 3) 模型 =========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = SimpleCNN().to(device)

# ========= 4) 损失函数 & 优化器 =========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========= 5) 训练与评估 =========
def train_one_epoch(epoch: int):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(test_loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

# ========= 6) 主流程 =========
def main():
    best_acc = 0.0
    start = time.time()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(epoch)
        te_loss, te_acc = evaluate()
        print(f"[EPOCH {epoch+1}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            print(f"[INFO] Saving model with new best accuracy: {te_acc:.2f}%")
            torch.save({
                "model_state": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch + 1
            }, CKPT_DIR / "mnist_cnn_best.pt")

    elapsed = time.time() - start
    print(f"[DONE] Best test acc: {best_acc:.2f}% | elapsed: {elapsed:.1f}s")
    print(f"[INFO] Best checkpoint: {CKPT_DIR / 'mnist_cnn_best.pt'}")

if __name__ == "__main__":
    main()
