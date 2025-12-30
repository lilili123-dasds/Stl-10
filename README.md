STL-10 优化器对比实验项目
项目概述
本项目旨在通过深度学习技术，在 STL-10 数据集上对比三种主流优化器（SGD、Adam、RMSprop）的性能表现。项目使用 MobileNetV2 作为基础模型，通过完整的训练、验证、测试流程，深入分析不同优化器在批次级别和轮次级别的表现差异，为优化器选择提供实验依据。

核心特性
特性	描述
多优化器对比	同时比较 SGD、Adam、RMSprop 三种优化器的性能
批次级别监控	记录每个批次的损失和准确率变化
全面的可视化	生成多种对比图表（批次对比、轮次对比、累积对比等）
自动保存机制	保存最佳训练和验证模型、每轮权重
详细的评估报告	生成混淆矩阵、分类报告、性能汇总
智能训练策略	实现 ReduceLROnPlateau 和 EarlyStopping 机制
文件结构
text
├── main.py                    # 主程序入口
├── data_loader.py            # 数据加载和预处理
├── model.py                  # 模型定义
├── train_eval.py             # 训练和评估函数
├── utils.py                  # 工具函数（如 EarlyStopping）
├── read_pth.py               # 模型权重文件分析工具
├── README.md                 # 项目说明文档
└── requirements.txt          # 依赖包列表（如有）
🔧 环境要求
主要依赖
Python 3.7+

PyTorch 1.9+

torchvision

scikit-learn

seaborn

matplotlib

numpy

安装依赖
bash
pip install torch torchvision scikit-learn seaborn matplotlib numpy
快速开始
1. 克隆项目
bash
git clone <项目地址>
cd <项目目录>
2. 运行主程序
bash
python main.py
3. 分析模型权重文件
bash
python read_pth.py
注意：需先修改 read_pth.py 中的 pth_file_path 为实际文件路径

配置参数
参数	默认值	描述
num_epochs	100	训练轮数
batch_size	32	批次大小
earlypatience	20	早停耐心值
device	自动检测	运行设备（cuda/cpu）
数据集
本项目使用 STL-10 数据集，包含10个类别：

类别编号	类别名称	训练样本数	测试样本数
1	airplane	500	800
2	bird	500	800
3	car	500	800
4	cat	500	800
5	deer	500	800
6	dog	500	800
7	horse	500	800
8	monkey	500	800
9	ship	500	800
10	truck	500	800
数据集特点：

总训练集：5,000张图片

总测试集：8,000张图片

图像大小：96×96像素

额外未标注数据：100,000张图片（本项目未使用）

优化器配置
优化器	学习率	动量	权重衰减	其他参数
SGD	0.005	0.9	5e-4	-
Adam	5e-5	-	1e-5	β1=0.9, β2=0.999
RMSprop	2e-4	-	2e-5	α=0.99
训练策略
数据增强
训练集：

随机裁剪 (224×224)

随机水平翻转 (p=0.5)

归一化 (ImageNet标准)

验证集/测试集：

尺寸调整 (224×224)

归一化 (ImageNet标准)

学习率调整
python
ReduceLROnPlateau(
    mode='min',      # 监控验证损失
    factor=0.5,      # 学习率衰减因子
    patience=3,      # 耐心值
    min_lr=1e-6,     # 最小学习率
    verbose=True     # 显示调整信息
)
早停机制
python
EarlyStopping(
    earlypatience=20,    # 早停耐心值
    delta=0,            # 最小改进阈值
    verbose=True        # 显示早停信息
)
输出文件结构
程序运行后会生成以下目录结构：

text
stl10_experiment/
├── stl10_checkpoints_epoch_100_val/     # 模型检查点
│   ├── best_train_w/                    # 最佳训练权重
│   │   ├── best_train_SGD_stl10.pth
│   │   ├── best_train_Adam_stl10.pth
│   │   └── best_train_RMSprop_stl10.pth
│   ├── best_val_w/                      # 最佳验证权重
│   │   ├── best_val_SGD_stl10.pth
│   │   ├── best_val_Adam_stl10.pth
│   │   └── best_val_RMSprop_stl10.pth
│   ├── sgd_epoch_100_val/               # SGD每轮权重
│   ├── adam_epoch_100_val/              # Adam每轮权重
│   ├── rmsprop_epoch_100_val/           # RMSprop每轮权重
│   └── test_results/                    # 详细测试结果
├── stl10_plots_epoch_100_val/           # 可视化图表
│   ├── batch_comparison_stl10_*.png     # 批次对比图
│   ├── epoch_comparison_stl10_*.png     # 轮次对比图
│   ├── cumulative_batch_comparison_*.png# 累积对比图
│   ├── performance_comparison_*.png     # 性能对比图
│   └── {优化器}_stl10_curves.png        # 单优化器曲线
├── stl10_confusion_matrices/            # 混淆矩阵
│   ├── SGD_confusion_matrix_*.png
│   ├── Adam_confusion_matrix_*.png
│   └── RMSprop_confusion_matrix_*.png
├── stl10_logs/                          # 训练日志
│   ├── SGD_*.log
│   ├── Adam_*.log
│   ├── RMSprop_*.log
│   └── stl10_experiment_summary.log     # 实验摘要
├── stl10_final_results.json             # 完整结果（JSON格式）
└── stl10_results_summary.txt            # 结果摘要
📊 可视化图表说明
1. 批次对比图
批次损失对比：三种优化器在批次级别的损失变化对比（平滑处理）

批次准确率对比：三种优化器在批次级别的准确率变化对比（平滑处理）

2. 轮次对比图（2×2布局）
位置	图表类型	描述
左上	训练损失对比	每个epoch的训练损失对比
右上	验证损失对比	每个epoch的验证损失对比
左下	训练准确率对比	每个epoch的训练准确率对比
右下	验证准确率对比	每个epoch的验证准确率对比
3. 累积批次对比图
累积批次损失平均值：批次损失的累积平均值变化

累积批次准确率平均值：批次准确率的累积平均值变化

4. 最佳性能对比图
左侧：最佳训练、验证、测试准确率对比

右侧：收敛速度对比（达到90%最佳验证准确率所需的epoch数）

5. 单优化器训练曲线图
每个优化器单独的训练损失和准确率曲线

6. 混淆矩阵
每个优化器在测试集上的混淆矩阵可视化，显示每类别的预测结果

结果分析指标
主要性能指标
指标	描述
训练准确率	模型在训练集上的表现
验证准确率	模型在验证集上的表现，用于模型选择
测试准确率	模型在测试集上的最终表现
批次稳定性	批次损失和准确率的标准差
收敛速度	达到最佳性能所需的时间/epoch数
输出结果格式
JSON格式：包含所有训练历史、批次历史、最终结果的完整数据

文本摘要：关键指标的文本汇总

分类报告：每个类别的精确率、召回率、F1分数

混淆矩阵：详细的预测结果矩阵

自定义修改指南
1. 更换数据集
修改 data_loader.py：

python
# 更改数据集
train_dataset = datasets.YourDataset(...)
test_dataset = datasets.YourDataset(...)
# 更新类别数量和名称
NUM_CLASSES = 你的类别数
2. 更换模型
修改 model.py：

python
def create_model(num_classes=10):
    model = models.你的模型(weights=True)  # 或 weights=None
    # 调整最后的分类层
    model.classifier = nn.Sequential(...)
    return model
3. 调整优化器参数
修改 main.py 中的 optimizers_config：

python
optimizers_config = {
    'SGD': lambda params: optim.SGD(params, lr=0.01, ...),
    'Adam': lambda params: optim.Adam(params, lr=0.001, ...),
    'RMSprop': lambda params: optim.RMSprop(params, lr=0.001, ...),
    # 添加新的优化器
    'Adagrad': lambda params: optim.Adagrad(params, lr=0.01, ...)
}
4. 修改训练参数
在 main.py 中调整：

python
num_epochs = 50      # 训练轮数
batch_size = 64      # 批次大小
earlypatience = 10   # 早停耐心值
故障排除
常见问题及解决方案
问题	可能原因	解决方案
内存不足	批次大小过大	减小 batch_size
训练速度慢	未使用GPU	检查CUDA是否可用，确保使用GPU
过拟合	模型复杂度过高	增加数据增强、权重衰减、早停
欠拟合	训练轮数不足	增加训练轮数、提高模型复杂度
梯度爆炸	学习率过高	降低学习率、使用梯度裁剪
日志分析
查看 stl10_logs/ 目录下的日志文件

每个优化器都有独立的日志文件

日志包含详细的训练过程和结果

扩展建议
1. 添加更多优化器
python
optimizers_config = {
    'SGD': ...,
    'Adam': ...,
    'RMSprop': ...,
    'Adagrad': optim.Adagrad,
    'Adadelta': optim.Adadelta,
    'AdamW': optim.AdamW,
    'Nadam': ...  # 需要自定义实现
}
2. 实现交叉验证
python
from sklearn.model_selection import KFold
# 实现K折交叉验证
3. 添加超参数优化
python
# 使用网格搜索
from sklearn.model_selection import GridSearchCV
# 或使用贝叶斯优化
4. 支持更多数据集
CIFAR-10/100

ImageNet

自定义数据集

参考文献
STL-10数据集论文
Coates, A., et al. (2011). "An Analysis of Single Layer Networks in Unsupervised Feature Learning"

MobileNetV2论文
Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

优化器对比研究
Ruder, S. (2016). "An overview of gradient descent optimization algorithms"

PyTorch官方文档
https://pytorch.org/docs/stable/index.html

许可证
本项目仅供学习和研究使用。如需商业用途，请遵守相关许可协议。

text
MIT License

Copyright (c) 2023 STL-10 Optimizer Comparison Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
致谢
感谢以下开源项目和库：

PyTorch 团队 - 提供强大的深度学习框架

torchvision 贡献者 - 提供预训练模型和数据集

STL-10 数据集创建者 - 提供高质量数据集

scikit-learn、seaborn、matplotlib - 提供数据分析和可视化工具

所有依赖库的维护者 - 确保项目的稳定运行

联系信息
如有问题或建议，请通过以下方式联系：

邮箱：[your-email@example.com]

GitHub Issues：[项目Issues页面]

项目主页：[项目GitHub页面]

温馨提示：
运行本项目前，请确保有足够的存储空间（约10GB用于数据集和模型保存）和计算资源（推荐使用GPU）。完整训练过程可能需要数小时，具体时间取决于硬件配置。祝您实验顺利！
