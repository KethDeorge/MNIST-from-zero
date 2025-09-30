# train_logreg_mnist.py
# 说明：
# - 使用最简单的线性模型（Softmax 回归），不含卷积/深层网络
# - 数据预处理仅 ToTensor()，避免归一化配置不当导致的不稳定
# - 优化器使用 SGD（学习率适中），训练过程非常稳定
# - 目标：快速得到 >90% 的测试准确率，确认整体链路 OK

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========= 1) 配置 =========
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5          # 5 个 epoch 就能到 90%+；想更高可调到 10
LR = 0.1            # 对于线性模型的稳定学习率；如不稳定可降到 0.05 或 0.01
DATA_ROOT = "./data"
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= 2) 数据集 & DataLoader =========
# 仅做 ToTensor：将 [0,255] 像素值转为 [0,1] 的 float32 Tensor
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

# CPU 训练不需要 pin_memory，保持默认 False 即可
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

# ========= 3) 模型（线性分类器：784 -> 10）=========
# 将 28x28 灰度图展平为 784 维，用一个线性层输出 10 类 logits，后接交叉熵损失即 Softmax 回归
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)  # 权重形状 [10, 784]，偏置 [10]

    def forward(self, x):
        x = x.view(x.size(0), -1)      # 展平为 [B, 784]
        return self.fc(x)               # 输出 [B, 10] 的 logits（未做 softmax，交叉熵内部会处理）

model = SoftmaxRegression().to(device)

# ========= 4) 损失函数 & 优化器 =========
criterion = nn.CrossEntropyLoss()                 # 自动做 LogSoftmax + NLLLoss
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# ========= 5) 训练与评估函数 =========
def train_one_epoch(epoch: int):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向
        logits = model(images)
        loss = criterion(logits, labels)

        # 反向与更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

# ========= 6) 训练主流程 =========
def main():
    best_acc = 0.0
    start = time.time()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(epoch)
        te_loss, te_acc = evaluate()
        print(f"[EPOCH {epoch+1}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.2f}%")

        # 线性模型参数很小，保存一次最好权重方便后续对比
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch + 1
            }, CKPT_DIR / "mnist_logreg_best.pt")

    elapsed = time.time() - start
    print(f"[DONE] Best test acc: {best_acc:.2f}% | elapsed: {elapsed:.1f}s")
    print(f"[INFO] Best checkpoint: {CKPT_DIR / 'mnist_logreg_best.pt'}")

if __name__ == "__main__":
    main()
