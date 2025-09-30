# visual_inspect.py
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# --------- 模型定义（与训练时保持一致） ---------
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64*14*14, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# --------- 可视化工具 ---------
def plot_confusion_matrix(y_true, y_pred, savepath=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    # 标注数字
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def show_grid(images, preds, labels, title, max_n=25, savepath=None):
    n = min(max_n, len(images))
    idxs = list(range(len(images)))
    random.shuffle(idxs)
    idxs = idxs[:n]
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    axes = np.array(axes).reshape(rows, cols)
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            i = idxs[k]
            img = images[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"p:{preds[i]} | y:{labels[i]}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def show_topk_bar(probs, label, pred, k=10, savepath=None):
    topk = torch.topk(probs, k)
    vals = topk.values.cpu().numpy()
    idxs = topk.indices.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(range(k), vals)
    ax.set_xticks(range(k))
    ax.set_xticklabels([str(i) for i in idxs])
    ax.set_ylim(0,1)
    ax.set_title(f"Top-{k} Probabilities | pred={pred} true={label}")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

# --------- 主流程 ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "cnn"], default="logreg")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--limit", type=int, default=None, help="limit number of test samples")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--invert", action="store_true", help="invert image (for black-on-white inputs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 与训练一致：这里默认只 ToTensor()，不做 Normalize（你的 logreg 脚本如此）
    tfms = [transforms.ToTensor()]
    if args.invert:
        # 如果你评估的是“黑字白底”的外部图片，可以选择性反相
        tfms.insert(0, transforms.RandomInvert(p=1.0))  # 仅演示；MNIST本身无需反相
    transform = transforms.Compose(tfms)

    test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
    if args.limit:
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 构建并加载模型
    if args.model == "logreg":
        model = SoftmaxRegression()
    else:
        model = SimpleCNN()
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)  # 兼容直接 state_dict 的情况
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_preds, all_labels = [], []
    all_images = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
            all_images.append(x.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images)

    acc = (all_preds == all_labels).mean() * 100.0
    print(f"[EVAL] Test accuracy: {acc:.2f}% on {len(all_labels)} samples")

    # 保存目录
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 混淆矩阵
    cm_path = str(save_dir / "confusion_matrix.png") if save_dir else None
    plot_confusion_matrix(all_labels, all_preds, savepath=cm_path)

    # 2) 随机样本网格（混合对/错）
    grid_all_path = str(save_dir / "grid_random.png") if save_dir else None
    show_grid(all_images, all_preds, all_labels, title="Random predictions", max_n=25, savepath=grid_all_path)

    # 3) 错误样本网格
    wrong_idx = np.where(all_preds != all_labels)[0]
    if len(wrong_idx) > 0:
        show_grid(all_images[wrong_idx], all_preds[wrong_idx], all_labels[wrong_idx],
                  title=f"Misclassified samples (n={len(wrong_idx)})", max_n=25,
                  savepath=(str(save_dir / "grid_wrong.png") if save_dir else None))
    else:
        print("[INFO] No misclassified samples—great!")

    # 4) 随机取 1 张画 top-10 概率柱状图
    i = random.randrange(len(all_labels))
    with torch.no_grad():
        x = all_images[i:i+1].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    topk_path = str(save_dir / "topk_bar.png") if save_dir else None
    show_topk_bar(probs, label=int(all_labels[i]), pred=int(all_preds[i]), k=10, savepath=topk_path)

if __name__ == "__main__":
    main()

