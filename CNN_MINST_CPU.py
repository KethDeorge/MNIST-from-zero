# -*- coding: utf-8 -*-
"""
MNIST CNN（PyTorch）——中文详尽注释版
-------------------------------------------------
功能：
1) 下载并加载 MNIST 数据集
2) 构建一个轻量级 CNN 模型（两层卷积 + 池化 + 两层全连接）
3) 训练若干轮，评估测试集精度
4) 保存测试精度最好的模型权重到 ./checkpoints/mnist_cnn_best.pt

提示：此版本在每个重要步骤都加了中文注释，便于新手理解。
"""

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ========= 1) 基础配置 =========
SEED = 42  # 固定随机种子，保证可复现：数据打乱、权重初始化等会更稳定
DATA_ROOT = "./data"  # MNIST 下载/缓存目录
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)  # 模型权重保存目录

# 可调参数（CPU 友好默认值）
BATCH_SIZE = 128        # 每个 batch 的样本数；越大内存占用越高
EPOCHS = 5              # 训练轮数；入门可先 1~5 轮验证流程
LR = 1e-3               # 学习率；Adam 的常用起点
USE_SUBSET = False      # True: 仅用子集（5k/1k）作快速实验；False: 使用完整 MNIST

# 线程（数据加载在 CPU 下可提速；根据机器配置适当调整）
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))    # 计算线程
TORCH_NUM_INTEROP = int(os.environ.get("TORCH_NUM_INTEROP", "4"))    # 线程间并行

# 设置线程与随机种子
torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP)
torch.manual_seed(SEED)

# 自动选择设备：优先使用 GPU（CUDA），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= 2) 数据集 & DataLoader =========
# transform 定义输入预处理：这里只做 ToTensor，将 [0,255] 像素转为浮点张量并缩放到 [0,1]
# 注：MNIST 的常见规范化是 Normalize((0.1307,), (0.3081,))，你也可以替换掉下面一行试试
transform = transforms.ToTensor()

# 下载/加载训练集与测试集（首次运行会自动下载）
train_ds = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

# 如果只想快速跑通流程，可启用子集（5k 训练样本 / 1k 测试样本）
if USE_SUBSET:
    train_ds = Subset(train_ds, range(0, 5000))
    test_ds  = Subset(test_ds,  range(0, 1000))

# DataLoader 负责批量读取、打乱数据以及多进程预取
NUM_WORKERS = 4  # 工作进程数；Windows 报错可改为 0；Linux/WSL 可尝试 4~6
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,    # 训练集需要 shuffle
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
test_loader  = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,   # 测试集不需要 shuffle
    num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2,
)
print(f"[INFO] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

# ========= 3) 模型（轻量 CNN，适合 CPU）=========
# 输入图像尺寸: [B, 1, 28, 28]
# conv1: 1->16  k=3 s=1 padding=0   28 -> 26  => 输出 [B, 16, 26, 26]
# conv2: 16->32 k=3 s=1 padding=0   26 -> 24  => 输出 [B, 32, 24, 24]
# pool:  2x2                       24 -> 12  => 输出 [B, 32, 12, 12]
# 展平：32*12*12 = 4608
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 1：从 1 个通道（灰度）提取 16 个特征通道
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        # 卷积层 2：在 16 个通道的基础上继续提取，得到 32 个特征通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # 最大池化：窗口 2x2，将特征图高宽减半
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Dropout：训练时随机丢弃一部分神经元，缓解过拟合
        self.drop = nn.Dropout(0.25)
        # 全连接层 1：卷积特征展平后映射到 128 维
        self.fc1 = nn.Linear(32*12*12, 128)
        # 全连接层 2（分类头）：128 -> 10 类，对应数字 0~9
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # ReLU 非线性激活，提升表达能力并引入非线性
        x = torch.relu(self.conv1(x))                   # [B, 16, 26, 26]
        x = self.pool(torch.relu(self.conv2(x)))        # [B, 32, 12, 12]
        x = self.drop(x)                                # 训练期随机屏蔽
        x = x.flatten(1)                                # 展平为 [B, 4608]
        x = torch.relu(self.fc1(x))                     # [B, 128]
        x = self.drop(x)
        return self.fc2(x)                              # [B, 10]（logits，未过 softmax）

# 实例化模型并放到设备上（CPU/GPU）
model = SimpleCNN().to(device)

# ========= 4) 损失函数 & 优化器 =========
# 交叉熵损失：适合多分类，输入要求是 logits（未归一化的分数）
criterion = nn.CrossEntropyLoss()
# Adam 优化器：自适应学习率，通常对初学者更稳
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========= 5) 训练与评估 =========

def train_one_epoch(epoch: int):
    """训练单个 epoch，返回训练集平均损失和准确率"""
    model.train()  # 切换到训练模式（启用 Dropout 等）
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)

        # 前向传播：得到 logits
        logits = model(images)
        loss = criterion(logits, labels)

        # 反向传播：清零梯度 -> 计算梯度 -> 更新参数
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True 可节省显存
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)              # 取每行最大值的索引作为预测类别
        correct += (preds == labels).sum().item() # 累加预测正确的样本数
        total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

@torch.no_grad()
def evaluate():
    """在测试集上评估，返回测试集平均损失和准确率"""
    model.eval()  # 切换到评估模式（关闭 Dropout 等）
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
    best_acc = 0.0  # 记录历史最好测试集准确率
    start = time.time()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(epoch)
        te_loss, te_acc = evaluate()
        print(f"[EPOCH {epoch+1}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.2f}%")

        # 如果当前测试精度优于历史最好，保存 checkpoint
        if te_acc > best_acc:
            best_acc = te_acc
            print(f"[INFO] Saving model with new best accuracy: {te_acc:.2f}%")
            torch.save({
                "model_state": model.state_dict(),  # 仅保存权重参数字典，便于后续加载
                "acc": best_acc,
                "epoch": epoch + 1
            }, CKPT_DIR / "mnist_cnn_best.pt")

    elapsed = time.time() - start
    print(f"[DONE] Best test acc: {best_acc:.2f}% | elapsed: {elapsed:.1f}s")
    print(f"[INFO] Best checkpoint: {CKPT_DIR / 'mnist_cnn_best.pt'}")


if __name__ == "__main__":
    main()
