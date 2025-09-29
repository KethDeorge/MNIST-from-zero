import torch
from torchvision import datasets, transforms

# 定义数据预处理：把图片转成 Tensor，并归一化到 [0,1]
transform = transforms.ToTensor()

# 下载训练集
train_dataset = datasets.MNIST(
    root="./data",       # 数据存放路径
    train=True,          # 训练集
    download=True,       # 如果本地没有则下载
    transform=transform
)

# 下载测试集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,         # 测试集
    download=True,
    transform=transform
)

print("训练集大小:", len(train_dataset))
print("测试集大小:", len(test_dataset))
