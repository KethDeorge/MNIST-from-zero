STILL IN BUILD 还在开发！

# 从零开始的手写识别（MNIST）
这真的是我作为一个新手写的最详细的project了，新手做项目若有问题多多包涵。

本项目运行在`fedora42`，默认会基本的linux操作和命令行基础。

注册github我就不交了。
## 概念部分
### 什么是神经网络/机器学习
本质上来说，我们要构建的就是一个函数$F(\vec{x})=\vec{y}$，就是通过输入一个东西来得到另一个东西，其中输入输出和函数都是自己定义。

比如我们这个项目就是输入一张图片，然后输出一个0-9的数字。而我们所要做的就是通过求得$F(x)$，使得输出的数字就是我们输入的那个数字的手写图片

而神经网络就是一种模拟人脑角色的过程，比如说我们会发现
- 数字7有一个折但是1没有折
- 数字3有两个上下排列的半圆，2只有一个

这就意味着
- 如果一个输入的图片有折，那么他是7的概率就是大于1的概率
- 如果有两个半圆，3的概率大于2

通过让机器不断的识别不同涂层的内容，来提取（机器认为的）特征，最后得出对于每一个数字可能性的最大答案，然后通过`归一化`来让所有的可能性统一的变为0-1的概率，最后得出最大的可能性的结论

卷积来说就是变得支持旋转和平移了（也可以让他不支持平移）。
### 什么是MNIST
MNIST是一组数据集，其中包含了很多的图片，每一张图片具有如下的特征：
- 内容只有手写的0-9
- 大写是28\*28的像素
- 单通道的灰度（i.e.只有黑白灰，没有彩色）

#### 为什么用MNIST
- 数据小，容易下载
- 适合作为新手开始第一次（卷积）神经网络
## 如何开始从零开始MNIST
### 建立环境
我们使用的是`python`环境，通过创建`venv`来构建虚拟环境，隔绝和主程序的环境，具体操作如下：
```zsh
# 确定python环境
python --version

# 创建一个叫做venv的虚拟环境
python -m venv venv

# 激活
source venv/bin/activate
```

在 Python 命令里，-m 的意思是“把某个库模块当作脚本来运行”。

在激活后会看到命令行出现了`<venv>`字样

接下来安装库：写文档：名字叫做`requirements.txt`
```zsh
# 基础科学计算
numpy
scipy
pandas
matplotlib

# 深度学习框架（CPU 版本）
torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu

# MNIST 数据集常用库
scikit-learn
tqdm

```

然后运行：
`pip install -r requirements.txt`

或者直接
```zsh
# 安装科学计算 & 常用库
pip install numpy scipy pandas matplotlib scikit-learn tqdm pillow

# 安装与 Python 3.13 兼容的 PyTorch CPU 版本及匹配的 torchvision
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
```
### 初始化git
本项目托管在github上面，所以也会教学最基础的关于github的操作。

由于我本人是一个不折不扣（并不是）的CLI派（有时候也会用`lazygit`），所以我会主要教学在命令行的操作

**下面的操作是我一般写项目的时候会用的，而不是说如何从github上面下载**
1. 在github上创建仓库
2. 在本地仓库建立和你的github的SSH连接
3. 开始连接

接下来开始说每一个的操作
#### github创建仓库
直接贴图，大家可能都知道怎么做

#### 开始让本地和github连接
确认SSH配置
`ls ~/.ssh/id_rsa.pub`如果不存在就生成SSH,生成代码自己在网上查看。
测试：
`ssh -T git@github.com`
如果是`Hi`开头说明可以连接

打开对应的文件夹
```zsh
# 建立新仓库
git init

#添加远程仓库地址
git remote add origin git@github.com:KethDeorge/MNIST-from-zero.git
```
检查：
`git remote -v`，如果看到就代表正常

#### git上传操作
上传代码如下：
```
git add .

git commit -m "Loarm"

git push
```
分别是添加，添加备注，上传。
也可以在lazygit里面按下a，c（输入注释），P（大写）上传。
第一次应该是这样：
**注意全部都在第二个区域操作**

### 下载数据集
代码如下：
```
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

```

# 附录QA
Q：如果git初始不是`main`怎么办？

A：
`git config --global init.defaultBranch main`

Q：py库安装失败
A：注意python版本

Q：不会生成SSH
A：`ssh-keygen -t rsa -b 4096 -C "你的GitHub邮箱"`
同时在cli建议使用`SSH`而不是`HTTPS`
