import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ========= 1) 配置 =========
SEED = 42
BATCH_SIZE = 128
EPOCHS = 1           # 只跑 1 个 epoch
LR = 1e-3            # Adam 学习率
DATA_ROOT = "./data"
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= 2) 数据集 & DataLoader =========
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

# 使用数据子集：5000 训练样本，1000 测试样本
train_ds = Subset(train_ds, range(0, 5000))
test_ds = Subset(test_ds, range(0, 1000))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2)

print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

# ========= 3) 模型（简化版 CNN）=========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),  # 1->16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # 16->32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                           # -> [B,32,12,12]
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*12*12, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)     # 输出 10 类别
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x

model = SimpleCNN().to(device)

# ========= 4) 损失函数 & 优化器 =========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========= 5) 训练与评估函数 =========
def train_one_epoch(epoch: int):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
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
    for images, labels in tqdm(test_loader, desc="Eval"):
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

        if te_acc > best_acc:
            best_acc = te_acc
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
