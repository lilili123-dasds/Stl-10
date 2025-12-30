import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import models, datasets
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from datetime import datetime
import json
from data_loader import get_data_loaders
from model import create_model
from train_eval import train_one_epoch, evaluate
from utils import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ================== 设置日志系统 ==================
def setup_logger(optimizer_name):
    """为每个优化器创建独立的日志文件"""
    log_dir = "stl10_logs"  # 日志文件夹名称
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{optimizer_name}_{timestamp}.log")
    
    logger = logging.getLogger(f"{optimizer_name}_logger")
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器，避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

# ================== 修改的train_one_epoch函数，返回批次损失和准确率 ==================
def train_one_epoch_with_batch_history(model, train_loader, criterion, optimizer, device):
    """训练一个epoch并返回批次损失和准确率历史"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # 用于存储批次历史的列表
    batch_loss_history = []
    batch_acc_history = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 统计
        _, preds = torch.max(outputs, 1)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples += batch_size
        
        # 记录每个批次的损失和准确率
        batch_loss = loss.item()
        batch_acc = torch.sum(preds == labels.data).item() / batch_size * 100
        batch_loss_history.append(batch_loss)
        batch_acc_history.append(batch_acc)
        
        # 每10个批次打印一次
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%')
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples * 100
    
    return epoch_loss, epoch_acc.item(), batch_loss_history, batch_acc_history

# ================== 创建混淆矩阵 ==================
def create_confusion_matrix(model, data_loader, device, class_names=None):
    """创建并保存混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 创建可视化
    plt.figure(figsize=(12, 10))  # STL-10有10个类别，需要更大的图
    if class_names is None:
        # STL-10的10个类别名称
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                       'dog', 'horse', 'monkey', 'ship', 'truck']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('STL-10 Dataset Confusion Matrix')  # 修改标题
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)  # x轴标签旋转45度，避免重叠
    plt.yticks(rotation=45)  # y轴标签旋转45度
    plt.tight_layout()
    
    return cm, all_preds, all_labels

def save_confusion_matrix_plot(optimizer_name, cm, test_acc, folder='stl10_confusion_matrices'):
    """保存混淆矩阵图片"""
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f'{optimizer_name}_confusion_matrix_{timestamp}.png')
    
    # 在图中添加准确率信息
    plt.figtext(0.5, 0.01, f'STL-10 Test Accuracy: {test_acc:.2f}%',
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# ================== 绘制批次损失和准确率对比图 ==================
def plot_batch_comparison(results, plots_dir='stl10_plots_epoch_100_val'):
    """绘制三个优化器的批次损失和准确率对比图"""
    os.makedirs(plots_dir, exist_ok=True)
    
    # 创建批次对比图
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 批次损失对比
    ax1 = axes[0]
    colors = {'SGD': 'blue', 'Adam': 'green', 'RMSprop': 'red'}
    markers = {'SGD': 'o', 'Adam': 's', 'RMSprop': '^'}
    
    for opt_name, color in colors.items():
        if opt_name in results and 'batch_loss_history' in results[opt_name]:
            batch_losses = results[opt_name]['batch_loss_history']
            # 对批次损失进行平滑处理（移动平均）
            window_size = min(50, len(batch_losses) // 10)
            if window_size > 1:
                smoothed_losses = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
                x = np.arange(len(smoothed_losses))
                ax1.plot(x, smoothed_losses, color=color, label=f'{opt_name} (smoothed)', 
                        linewidth=2, alpha=0.8)
            else:
                x = np.arange(len(batch_losses))
                ax1.plot(x, batch_losses, color=color, label=opt_name, 
                        linewidth=1, alpha=0.6)
    
    ax1.set_title('STL-10 Dataset - Batch Loss Comparison (Smoothed)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Index')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数刻度更清晰显示
    
    # 批次准确率对比
    ax2 = axes[1]
    for opt_name, color in colors.items():
        if opt_name in results and 'batch_acc_history' in results[opt_name]:
            batch_accs = results[opt_name]['batch_acc_history']
            # 对批次准确率进行平滑处理
            window_size = min(50, len(batch_accs) // 10)
            if window_size > 1:
                smoothed_accs = np.convolve(batch_accs, np.ones(window_size)/window_size, mode='valid')
                x = np.arange(len(smoothed_accs))
                ax2.plot(x, smoothed_accs, color=color, label=f'{opt_name} (smoothed)', 
                        linewidth=2, alpha=0.8)
            else:
                x = np.arange(len(batch_accs))
                ax2.plot(x, batch_accs, color=color, label=opt_name, 
                        linewidth=1, alpha=0.6)
    
    ax2.set_title('STL-10 Dataset - Batch Accuracy Comparison (Smoothed)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Index')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'batch_comparison_stl10_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Batch comparison plot saved to: {plot_path}")
    return plot_path

# ================== 绘制每个epoch的Loss和Acc对比图 ==================
def plot_epoch_comparison(results, plots_dir='stl10_plots_epoch_100_val'):
    """绘制三个优化器的每个epoch的Loss和Acc对比图"""
    os.makedirs(plots_dir, exist_ok=True)
    
    # 创建epoch对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'SGD': 'blue', 'Adam': 'green', 'RMSprop': 'red'}
    markers = {'SGD': 'o', 'Adam': 's', 'RMSprop': '^'}
    line_styles = {'SGD': '-', 'Adam': '--', 'RMSprop': '-.'}
    
    # 获取最大epoch数
    max_epochs = 0
    for opt_name in results.keys():
        if 'train_losses' in results[opt_name]:
            max_epochs = max(max_epochs, len(results[opt_name]['train_losses']))
    
    # 1. 训练损失对比
    ax1 = axes[0, 0]
    for opt_name, color in colors.items():
        if opt_name in results and 'train_losses' in results[opt_name]:
            train_losses = results[opt_name]['train_losses']
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], marker=markers[opt_name], 
                    markersize=4, alpha=0.8)
    
    ax1.set_title('STL-10 Dataset - Epoch Train Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数刻度
    
    # 2. 验证损失对比
    ax2 = axes[0, 1]
    for opt_name, color in colors.items():
        if opt_name in results and 'val_losses' in results[opt_name]:
            val_losses = results[opt_name]['val_losses']
            epochs = range(1, len(val_losses) + 1)
            ax2.plot(epochs, val_losses, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], marker=markers[opt_name], 
                    markersize=4, alpha=0.8)
    
    ax2.set_title('STL-10 Dataset - Epoch Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 对数刻度
    
    # 3. 训练准确率对比
    ax3 = axes[1, 0]
    for opt_name, color in colors.items():
        if opt_name in results and 'train_accs' in results[opt_name]:
            train_accs = results[opt_name]['train_accs']
            epochs = range(1, len(train_accs) + 1)
            ax3.plot(epochs, train_accs, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], marker=markers[opt_name], 
                    markersize=4, alpha=0.8)
    
    ax3.set_title('STL-10 Dataset - Epoch Train Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])  # 准确率范围0-100%
    
    # 4. 验证准确率对比
    ax4 = axes[1, 1]
    for opt_name, color in colors.items():
        if opt_name in results and 'val_accs' in results[opt_name]:
            val_accs = results[opt_name]['val_accs']
            epochs = range(1, len(val_accs) + 1)
            ax4.plot(epochs, val_accs, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], marker=markers[opt_name], 
                    markersize=4, alpha=0.8)
    
    ax4.set_title('STL-10 Dataset - Epoch Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])  # 准确率范围0-100%
    
    plt.suptitle('STL-10 Dataset - Epoch-wise Performance Comparison of Optimizers', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'epoch_comparison_stl10_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Epoch comparison plot saved to: {plot_path}")
    return plot_path

# ================== 绘制累积批次对比图 ==================
def plot_cumulative_batch_comparison(results, plots_dir='stl10_plots_epoch_100_val'):
    """绘制累积的批次损失和准确率对比图"""
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'SGD': 'blue', 'Adam': 'green', 'RMSprop': 'red'}
    line_styles = {'SGD': '-', 'Adam': '--', 'RMSprop': '-.'}
    
    # 累积批次损失
    ax1 = axes[0]
    for opt_name, color in colors.items():
        if opt_name in results and 'batch_loss_history' in results[opt_name]:
            batch_losses = results[opt_name]['batch_loss_history']
            # 计算累积平均值
            cumulative_avg = np.cumsum(batch_losses) / (np.arange(len(batch_losses)) + 1)
            x = np.arange(len(cumulative_avg))
            ax1.plot(x, cumulative_avg, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], alpha=0.8)
    
    ax1.set_title('STL-10 - Cumulative Batch Loss Average', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Index')
    ax1.set_ylabel('Cumulative Average Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 累积批次准确率
    ax2 = axes[1]
    for opt_name, color in colors.items():
        if opt_name in results and 'batch_acc_history' in results[opt_name]:
            batch_accs = results[opt_name]['batch_acc_history']
            # 计算累积平均值
            cumulative_avg = np.cumsum(batch_accs) / (np.arange(len(batch_accs)) + 1)
            x = np.arange(len(cumulative_avg))
            ax2.plot(x, cumulative_avg, color=color, label=opt_name, 
                    linewidth=2, linestyle=line_styles[opt_name], alpha=0.8)
    
    ax2.set_title('STL-10 - Cumulative Batch Accuracy Average', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Index')
    ax2.set_ylabel('Cumulative Average Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'cumulative_batch_comparison_stl10_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cumulative batch comparison plot saved to: {plot_path}")
    return plot_path

# ================== 绘制最佳性能对比图 ==================
def plot_best_performance_comparison(results, plots_dir='stl10_plots_epoch_100_val'):
    """绘制最佳训练和验证性能对比图"""
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'SGD': 'blue', 'Adam': 'green', 'RMSprop': 'red'}
    
    # 提取数据
    optimizers = []
    best_train_accs = []
    best_val_accs = []
    final_test_accs = []
    
    for opt_name in results.keys():
        if 'best_train_acc' in results[opt_name] and 'best_val_acc' in results[opt_name]:
            optimizers.append(opt_name)
            best_train_accs.append(results[opt_name]['best_train_acc'])
            best_val_accs.append(results[opt_name]['best_val_acc'])
            if 'final_test_acc' in results[opt_name]:
                final_test_accs.append(results[opt_name]['final_test_acc'])
    
    # 条形图宽度
    x = np.arange(len(optimizers))
    width = 0.25
    
    # 1. 训练和验证准确率对比
    ax1 = axes[0]
    ax1.bar(x - width, best_train_accs, width, label='Best Train Acc', color='blue', alpha=0.7)
    ax1.bar(x, best_val_accs, width, label='Best Val Acc', color='orange', alpha=0.7)
    if final_test_accs:
        ax1.bar(x + width, final_test_accs, width, label='Test Acc', color='green', alpha=0.7)
    
    ax1.set_title('STL-10 Dataset - Best Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(optimizers)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # 添加数值标签
    for i, (train_acc, val_acc) in enumerate(zip(best_train_accs, best_val_accs)):
        ax1.text(i - width, train_acc + 1, f'{train_acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i, val_acc + 1, f'{val_acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
        if final_test_accs:
            ax1.text(i + width, final_test_accs[i] + 1, f'{final_test_accs[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
    
    # 2. 收敛速度对比（达到最佳验证准确率的epoch数）
    ax2 = axes[1]
    convergence_epochs = []
    for opt_name in optimizers:
        if 'val_accs' in results[opt_name]:
            val_accs = results[opt_name]['val_accs']
            best_val_acc = max(val_accs)
            # 找到达到90%最佳性能的epoch
            target_acc = best_val_acc * 0.9
            convergence_epoch = None
            for epoch, acc in enumerate(val_accs):
                if acc >= target_acc:
                    convergence_epoch = epoch + 1
                    break
            if convergence_epoch is None:
                convergence_epoch = len(val_accs)
            convergence_epochs.append(convergence_epoch)
    
    bars = ax2.bar(x, convergence_epochs, width=0.5, 
                  color=[colors[opt] for opt in optimizers], alpha=0.7)
    
    ax2.set_title('STL-10 Dataset - Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Epochs to reach 90% of best val acc')
    ax2.set_xticks(x)
    ax2.set_xticklabels(optimizers)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, epoch in zip(bars, convergence_epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{epoch}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('STL-10 Dataset - Optimizer Performance Summary', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'performance_comparison_stl10_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison plot saved to: {plot_path}")
    return plot_path

# ================== 保存结果到JSON ==================
def save_results_to_json(results, filename='stl10_experiment_results.json'):
    """保存所有结果到JSON文件"""
    # 转换numpy数组为列表
    results_dict = {}
    for opt_name, data in results.items():
        results_dict[opt_name] = {
            'train_losses': [float(x) for x in data['train_losses']],
            'train_accs': [float(x) for x in data['train_accs']],
            'val_losses': [float(x) for x in data['val_losses']],
            'val_accs': [float(x) for x in data['val_accs']],
            'batch_loss_history': [float(x) for x in data.get('batch_loss_history', [])],
            'batch_acc_history': [float(x) for x in data.get('batch_acc_history', [])],
            'best_train_acc': float(data['best_train_acc']),
            'best_val_acc': float(data['best_val_acc']),
            'final_test_acc': float(data['final_test_acc']),
            'final_test_loss': float(data.get('final_test_loss', 0)),
            'confusion_matrix': data.get('confusion_matrix', []).tolist() if 'confusion_matrix' in data else [],
            'classification_report': data.get('classification_report', {})
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    return filename

# ================== STL-10 特定配置 ==================
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 
                 'dog', 'horse', 'monkey', 'ship', 'truck']
NUM_CLASSES = 10  # STL-10有10个类别

# ================== 主程序 ==================
# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
num_epochs = 100  # 训练轮数
batch_size = 32  # 批大小
earlypatience = 20  # 早停耐心值

# 创建目录
plots_dir = 'stl10_plots_epoch_100_val'
os.makedirs(plots_dir, exist_ok=True)
confusion_dir = 'stl10_confusion_matrices'
os.makedirs(confusion_dir, exist_ok=True)
logs_dir = 'stl10_logs'
os.makedirs(logs_dir, exist_ok=True)

# ================== 创建保存目录 ==================
base_dir = "stl10_checkpoints_epoch_100_val"
os.makedirs(base_dir, exist_ok=True)

best_train_w_folder = os.path.join(base_dir, "best_train_w")
os.makedirs(best_train_w_folder, exist_ok=True)

best_val_w_folder = os.path.join(base_dir, "best_val_w")
os.makedirs(best_val_w_folder, exist_ok=True)

epoch_folders = {
    'SGD': os.path.join(base_dir, "sgd_epoch_100_val"),
    'Adam': os.path.join(base_dir, "adam_epoch_100_val"),
    'RMSprop': os.path.join(base_dir, "rmsprop_epoch_100_val")
}
for folder in epoch_folders.values():
    os.makedirs(folder, exist_ok=True)

# 加载STL-10数据
train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)

# 定义优化器配置
optimizers_config = {
    'SGD': lambda params: optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4),
    'Adam': lambda params: optim.Adam(params, lr=5e-5, weight_decay=1e-5),
    'RMSprop': lambda params: optim.RMSprop(params, lr=2e-4, alpha=0.99, weight_decay=2e-5)
}

results = {}

for opt_name in optimizers_config:
    print(f"\n{'='*60}")
    print(f"Training STL-10 Dataset with optimizer: {opt_name}")
    print(f"{'='*60}")
    
    # 设置日志
    logger, log_filename = setup_logger(opt_name)
    logger.info(f"STL-10 Dataset Training with {opt_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    
    # 创建模型
    model = create_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers_config[opt_name](model.parameters())
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6,verbose=True)
    early_stopping = EarlyStopping(
        earlypatience=earlypatience, 
        verbose=True, 
        path=os.path.join(best_val_w_folder, f'{opt_name}_stl10_checkpoint.pt')
    )
    
    best_train_acc = 0.0
    best_val_acc = 0.0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    batch_loss_history = []  # 存储所有批次的损失
    batch_acc_history = []   # 存储所有批次的准确率
    
    logger.info(f"Model architecture: MobileNetV2")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"STL-10 Classes: {STL10_CLASSES}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc, epoch_batch_losses, epoch_batch_accs = train_one_epoch_with_batch_history(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # 保存批次历史
        batch_loss_history.extend(epoch_batch_losses)
        batch_acc_history.extend(epoch_batch_accs)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 日志记录
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 保存权重
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            train_save_path = os.path.join(best_train_w_folder, f'best_train_{opt_name}_stl10.pth')
            torch.save(model.state_dict(), train_save_path)
            logger.info(f"New best TRAIN accuracy ({best_train_acc:.2f}%) - saved to {train_save_path}")
            print(f"     New best TRAIN accuracy ({best_train_acc:.2f}%)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            val_save_path = os.path.join(best_val_w_folder, f'best_val_{opt_name}_stl10.pth')
            torch.save(model.state_dict(), val_save_path)
            logger.info(f"New best VALIDATION accuracy ({best_val_acc:.2f}%) - saved to {val_save_path}")
            print(f"     New best VALIDATION accuracy ({best_val_acc:.2f}%)")
        
        epoch_save_path = os.path.join(epoch_folders[opt_name], f'{opt_name}_epoch_{epoch+1}_stl10.pth')
        torch.save(model.state_dict(), epoch_save_path)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            print("Early stopping")
            break
    
    # 保存批次历史到结果
    results[opt_name] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'batch_loss_history': batch_loss_history,
        'batch_acc_history': batch_acc_history,
        'best_train_acc': best_train_acc,
        'best_val_acc': best_val_acc,
        'train_epochs': len(train_losses)
    }
    
    # 训练曲线图（单个优化器）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title(f'STL-10 Dataset - {opt_name} - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='orange')
    plt.title(f'STL-10 Dataset - {opt_name} - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{opt_name}_stl10_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"STL-10 training curves saved to {plot_path}")
    
    logger.info(f"STL-10 training completed for {opt_name}")
    logger.info(f"Best train accuracy: {best_train_acc:.2f}%")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Total training epochs: {len(train_losses)}")
    logger.info(f"Total batches: {len(batch_loss_history)}")
    logger.info(f"Log file saved to: {log_filename}")

# ================== 绘制对比图 ==================
print("\n" + "="*60)
print("Generating comparison plots...")
print("="*60)

# 绘制批次对比图
batch_comparison_path = plot_batch_comparison(results, plots_dir)

# 绘制epoch对比图
epoch_comparison_path = plot_epoch_comparison(results, plots_dir)

# 绘制累积批次对比图
cumulative_path = plot_cumulative_batch_comparison(results, plots_dir)

# ================== 在测试集上评估 ==================
print("\n" + "="*60)
print("STL-10 DATASET - FINAL EVALUATION ON TEST SET")
print("="*60)

for opt_name in results.keys():
    print(f"\nEvaluating {opt_name} on STL-10 test set...")
    
    # 加载该优化器的最佳验证集模型
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model_path = os.path.join(best_val_w_folder, f'best_val_{opt_name}_stl10.pth')
    model.load_state_dict(torch.load(model_path))
    
    # 设置日志
    logger, _ = setup_logger(opt_name + "_test")
    logger.info(f"STL-10 Testing {opt_name} model loaded from {model_path}")
    logger.info(f"Best validation accuracy: {results[opt_name]['best_val_acc']:.2f}%")
    
    # 在测试集上评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 创建混淆矩阵
    cm, all_preds, all_labels = create_confusion_matrix(
        model, test_loader, device, class_names=STL10_CLASSES
    )
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, 
                                  target_names=STL10_CLASSES,
                                  output_dict=True)
    
    # 保存混淆矩阵图片
    cm_path = save_confusion_matrix_plot(opt_name, cm, test_acc, folder=confusion_dir)
    
    # 保存结果
    results[opt_name]['final_test_loss'] = test_loss
    results[opt_name]['final_test_acc'] = test_acc
    results[opt_name]['confusion_matrix'] = cm
    results[opt_name]['classification_report'] = report
    
    # 日志记录
    logger.info(f"STL-10 Test Loss: {test_loss:.4f}")
    logger.info(f"STL-10 Test Accuracy: {test_acc:.2f}%")
    logger.info(f"STL-10 Confusion matrix saved to {cm_path}")
    
    # 打印分类报告摘要
    report_str = classification_report(all_labels, all_preds, target_names=STL10_CLASSES)
    logger.info(f"STL-10 Classification Report:\n{report_str}")
    
    print(f"{opt_name}:")
    print(f"  - Final Test Loss: {test_loss:.4f}")
    print(f"  - Final Test Accuracy: {test_acc:.2f}%")
    print(f"  - Best Validation Accuracy: {results[opt_name]['best_val_acc']:.2f}%")
    print(f"  - Confusion matrix saved to: {cm_path}")
    
    # 保存详细测试结果
    test_results_folder = os.path.join(base_dir, "test_results")
    os.makedirs(test_results_folder, exist_ok=True)
    
    # 保存文本结果
    result_file = os.path.join(test_results_folder, f'{opt_name}_stl10_test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"STL-10 DATASET TEST RESULTS - {opt_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: STL-10 (10 classes)\n")
        f.write(f"Model: MobileNetV2\n")
        f.write(f"Optimizer: {opt_name}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Training epochs: {results[opt_name].get('train_epochs', 'N/A')}\n")
        f.write(f"Total batches: {len(results[opt_name].get('batch_loss_history', []))}\n")
        f.write(f"Best Train Accuracy: {results[opt_name]['best_train_acc']:.2f}%\n")
        f.write(f"Best Validation Accuracy: {results[opt_name]['best_val_acc']:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(f"STL-10 Classes: {STL10_CLASSES}\n\n")
        
        # 批次统计
        if 'batch_loss_history' in results[opt_name]:
            batch_losses = results[opt_name]['batch_loss_history']
            batch_accs = results[opt_name]['batch_acc_history']
            f.write("Batch Statistics:\n")
            f.write(f"  Min batch loss: {min(batch_losses):.6f}\n")
            f.write(f"  Max batch loss: {max(batch_losses):.6f}\n")
            f.write(f"  Avg batch loss: {np.mean(batch_losses):.6f}\n")
            f.write(f"  Min batch acc: {min(batch_accs):.2f}%\n")
            f.write(f"  Max batch acc: {max(batch_accs):.2f}%\n")
            f.write(f"  Avg batch acc: {np.mean(batch_accs):.2f}%\n\n")
        
        # 计算每个类别的准确率
        f.write("Per-class accuracy from confusion matrix:\n")
        for i, class_name in enumerate(STL10_CLASSES):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum() * 100
                f.write(f"  {class_name:10s}: {class_acc:6.2f}% ({cm[i, i]}/{cm[i].sum()})\n")
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_str)
    
    print(f"  - Detailed results saved to: {result_file}")

# ================== 绘制性能对比图 ==================
print("\n" + "="*60)
print("Generating performance comparison plots...")
print("="*60)

# 绘制最佳性能对比图
performance_comparison_path = plot_best_performance_comparison(results, plots_dir)

# ================== 保存所有结果 ==================
print("\n" + "="*60)
print("STL-10 DATASET - ALL FINAL TEST RESULTS SUMMARY")
print("="*60)

# 创建汇总表格
print("\n" + "-"*90)
print(f"{'Optimizer':<12} {'Best Val Acc':<15} {'Test Acc':<15} {'Test Loss':<12} {'Epochs':<10} {'Batches':<10}")
print("-"*90)

best_test_acc = 0
best_optimizer = ""

for opt_name in results.keys():
    val_acc = results[opt_name]['best_val_acc']
    test_acc = results[opt_name]['final_test_acc']
    test_loss = results[opt_name]['final_test_loss']
    epochs = results[opt_name].get('train_epochs', num_epochs)
    batches = len(results[opt_name].get('batch_loss_history', []))
    
    print(f"{opt_name:<12} {val_acc:<15.2f}% {test_acc:<15.2f}% {test_loss:<12.4f} {epochs:<10} {batches:<10}")
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_optimizer = opt_name

print("-"*90)
print(f"\nBest performing optimizer on STL-10: {best_optimizer} with {best_test_acc:.2f}% test accuracy")

# 保存所有结果到JSON文件
json_path = save_results_to_json(results, filename='stl10_final_results.json')
print(f"\nAll STL-10 results saved to JSON: {json_path}")

# ================== 创建汇总日志 ==================
summary_logger = logging.getLogger("stl10_summary_logger")
summary_logger.setLevel(logging.INFO)

summary_log_path = os.path.join(logs_dir, "stl10_experiment_summary.log")
summary_handler = logging.FileHandler(summary_log_path, encoding='utf-8')
summary_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-d %H:%M:%S'))
summary_logger.addHandler(summary_handler)

summary_logger.info("="*70)
summary_logger.info("STL-10 DATASET - EXPERIMENT SUMMARY")
summary_logger.info("="*70)
summary_logger.info(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_logger.info(f"Device: {device}")
summary_logger.info(f"Batch Size: {batch_size}")
summary_logger.info(f"Model: MobileNetV2")
summary_logger.info(f"Number of Classes: {NUM_CLASSES}")
summary_logger.info(f"Batch Comparison Plot: {batch_comparison_path}")
summary_logger.info(f"Epoch Comparison Plot: {epoch_comparison_path}")
summary_logger.info(f"Cumulative Batch Plot: {cumulative_path}")
summary_logger.info(f"Performance Comparison Plot: {performance_comparison_path}")
summary_logger.info("-"*70)

for opt_name in results.keys():
    val_acc = results[opt_name]['best_val_acc']
    test_acc = results[opt_name]['final_test_acc']
    epochs = results[opt_name].get('train_epochs', num_epochs)
    batches = len(results[opt_name].get('batch_loss_history', []))
    
    summary_logger.info(f"{opt_name:10s} | Val Acc: {val_acc:6.2f}% | "
                       f"Test Acc: {test_acc:6.2f}% | Epochs: {epochs:3d} | "
                       f"Batches: {batches:5d}")

summary_logger.info("-"*70)
summary_logger.info(f"Best optimizer: {best_optimizer} ({best_test_acc:.2f}%)")
summary_logger.info("="*70)
summary_logger.info(f"Results saved to: {json_path}")
summary_logger.info(f"Logs directory: {logs_dir}/")
summary_logger.info(f"Plots directory: {plots_dir}/")
summary_logger.info(f"Comparison plots generated")

# 保存简化的结果摘要
summary_file = "stl10_results_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("STL-10 Dataset - Optimization Experiment Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Experiment conducted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: MobileNetV2\n")
    f.write(f"Number of classes: {NUM_CLASSES}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Max epochs: {num_epochs}\n")
    f.write(f"Batch comparison plot: {batch_comparison_path}\n")
    f.write(f"Epoch comparison plot: {epoch_comparison_path}\n")
    f.write(f"Cumulative batch plot: {cumulative_path}\n")
    f.write(f"Performance comparison plot: {performance_comparison_path}\n\n")
    
    f.write("Results Summary:\n")
    f.write("-"*60 + "\n")
    f.write(f"{'Optimizer':<12} {'Val Acc':<10} {'Test Acc':<10} {'Epochs':<8} {'Batches':<10}\n")
    f.write("-"*60 + "\n")
    
    for opt_name in results.keys():
        val_acc = results[opt_name]['best_val_acc']
        test_acc = results[opt_name]['final_test_acc']
        epochs = results[opt_name].get('train_epochs', num_epochs)
        batches = len(results[opt_name].get('batch_loss_history', []))
        f.write(f"{opt_name:<12} {val_acc:<10.2f}% {test_acc:<10.2f}% {epochs:<8} {batches:<10}\n")
    
    f.write("-"*60 + "\n")
    f.write(f"\nBest optimizer: {best_optimizer} ({best_test_acc:.2f}% test accuracy)\n")
    f.write(f"\nBatch Analysis:\n")
    
    for opt_name in results.keys():
        if 'batch_loss_history' in results[opt_name]:
            batch_losses = results[opt_name]['batch_loss_history']
            batch_accs = results[opt_name]['batch_acc_history']
            f.write(f"\n{opt_name}:\n")
            f.write(f"  - Average batch loss: {np.mean(batch_losses):.6f}\n")
            f.write(f"  - Average batch accuracy: {np.mean(batch_accs):.2f}%\n")
            f.write(f"  - Batch loss std: {np.std(batch_losses):.6f}\n")
            f.write(f"  - Batch accuracy std: {np.std(batch_accs):.2f}%\n")
    
    f.write(f"\nDetailed results in: {json_path}\n")
    f.write(f"Logs in: {logs_dir}/\n")
    f.write(f"Plots in: {plots_dir}/\n")

print("\n" + "="*70)
print("STL-10 DATASET EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"Summary of directories created:")
print(f"  • Logs:              {logs_dir}/")
print(f"  • Training plots:    {plots_dir}/")
print(f"  • Confusion matrices:{confusion_dir}/")
print(f"  • Model checkpoints: {base_dir}/")
print(f"  • Test results:      {base_dir}/test_results/")
print(f"\nKey files generated:")
print(f"  • Full results (JSON): {json_path}")
print(f"  • Results summary:     {summary_file}")
print(f"  • Experiment summary:  {summary_log_path}")
print(f"\nComparison plots generated:")
print(f"  • Batch comparison:       {batch_comparison_path}")
print(f"  • Epoch comparison:       {epoch_comparison_path}")
print(f"  • Cumulative batch:       {cumulative_path}")
print(f"  • Performance comparison: {performance_comparison_path}")
print(f"\nBatch analysis completed for all optimizers")
print(f"Best optimizer on STL-10: {best_optimizer} ({best_test_acc:.2f}% test accuracy)")
print("="*70)