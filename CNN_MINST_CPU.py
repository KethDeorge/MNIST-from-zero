import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ========= 1) 基础配置 =========
SEED = 42
DATA_ROOT = "./data"
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 可调参数（先给出适合 CPU 的默认值）
BATCH_SIZE = 128
EPOCHS = 5                # 小机子 CPU 也能承受；先 5，想更快先改成 1
LR = 1e-3                 # Adam 默认学习率
USE_SUBSET = False        # True: 先用 5k/1k 快速验证；False: 全量 MNIST

# 线程（根据你 4 大核 8 线程的 CPU，经验值 4~6 比较稳）
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
TORCH_NUM_INTEROP = int(os.environ.get("TORCH_NUM_INTEROP", "4"))

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= 2) 数据集 & DataLoader =========
# 仅 ToTensor()，也可切换到 Normalize((0.1307,), (0.3081,))
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

if USE_SUBSET:
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, range(0, 5000))
    test_ds  = Subset(test_ds,  range(0, 1000))

# DataLoader：CPU 环境建议开点 worker 提升数据吞吐
NUM_WORKERS = 4      # 你的机器可尝试 4 或 6，看整体负载
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
test_loader  = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

# ========= 3) 模型（轻量 CNN，适合 CPU）=========
# 输入: [B,1,28,28]
# conv1: 1->16  k=3 -> 28->26
# conv2: 16->32 k=3 -> 26->24
# pool: 2x2     -> 24->12
# 展平: 32*12*12 = 4608
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里的层结构与你的原始代码保持一致
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 32 filters
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32*12*12, 128)  # 128 units
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # conv1
        x = self.pool(torch.relu(self.conv2(x)))  # conv2
        x = self.drop(x)
        x = x.flatten(1)  # 展平
        x = torch.relu(self.fc1(x))  # fc1
        x = self.drop(x)
        return self.fc2(x)  # 输出 logits

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

        optimizer.zero_grad(set_to_none=True)  # 更省内存的清梯度
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(test_loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

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

        # 保存模型时，检查准确率是否有更新
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
