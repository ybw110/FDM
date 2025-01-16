import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from ybw_single_model import *
from ybw_dataloader import get_data_loaders, set_seed
import ybw_config as Config
import seaborn as sns

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', colour='RED', ncols=120)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.argmax(1)).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_accuracy = 100. * correct / total
    return avg_loss, avg_accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc='Validation', ncols=120)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels.argmax(1))  # 修改这里以匹配您的标签格式
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(1)).sum().item()  # 修改这里以匹配您的标签格式

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(12, 10))

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})

    # 设置两个y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制损失曲线
    ax1.plot(range(epochs), train_losses, color='blue', linestyle='-', linewidth=3, label='Train Loss')
    ax1.plot(range(epochs), val_losses, color='blue', linestyle='--', linewidth=3, label='Val Loss')

    # 绘制准确率曲线
    ax2.plot(range(epochs), train_accuracies, color='orange', linestyle='-', linewidth=3, label='Train Accuracy')
    ax2.plot(range(epochs), val_accuracies, color='orange', linestyle='--', linewidth=3, label='Val Accuracy')

    # 设置x轴标签
    ax1.set_xlabel('Number of Epochs', fontsize=14)

    # 设置y轴标签和范围
    ax1.set_ylabel('Loss', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)

    # 设置y轴的范围（根据实际数据调整）
    ax1.set_ylim(0, max(max(train_losses), max(val_losses)))
    ax2.set_ylim(80, 100)  # 假设准确率在80%到100%之间

    # 设置刻度标签的字体大小
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=12)

    # 添加网格
    plt.grid(True, linestyle=':', alpha=0.5)

    # 设置标题
    plt.title('Training and Validation Metrics', fontsize=16)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(results_dir, 'loss_accuracy_plot.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', normalize=True):
    plt.figure(figsize=(12, 10))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    # 使用seaborn来绘制更美观的热力图
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True)

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    # 调整x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # 调整布局，确保所有元素都能显示
    plt.tight_layout()
    # 保存高质量图片
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 创建基于日期和时间的结果目录
# 选择对应的前缀
author_name = 'Saluja'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 将作者名字作为前缀加入文件夹命名
results_dir = os.path.join('single_results', f"{author_name}_{current_time}")
os.makedirs(results_dir, exist_ok=True)

def main():
    set_seed(Config.seed)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    train_loader, val_loader, test_loader = get_data_loaders(
        Config.csv_file, Config.root_dir, Config.angle,
        batch_size=Config.batch_size,
        train_ratio=Config.train_ratio,
        val_ratio=Config.val_ratio,
        test_ratio=Config.test_ratio,
        num_workers=Config.num_workers
    )
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Testing on {len(test_loader.dataset)} samples")

    model = DetectionCNN(num_classes=5,dropout_rate=Config.dropout_rate).to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=Config.factor, patience=Config.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_accuracy = 0
    best_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(Config.epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                                                     Config.epochs)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{Config.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}/{Config.epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_val_loss = val_loss
            model_filename = f"epoch={best_epoch:02d}-val_loss={best_val_loss:.2f}-val_acc={best_val_accuracy:.2f}.pth"
            torch.save(model.state_dict(), os.path.join(results_dir, model_filename))
            print(f'Saved best model: {model_filename}')

        scheduler.step(val_loss)
    print("Training Complete.")

    print("-------------------------------------------------")
    print("Train Loss: ", np.mean(train_losses))
    print("Train Accuracy: ", np.mean(train_accuracies))
    print("Val Loss: ", np.mean(val_losses))
    print("Val Accuracy: ", np.mean(val_accuracies))

    # 绘制训练和验证的损失
    # 在训练循环结束后
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, Config.epochs)

    # Test the best model
    best_model_path = os.path.join(results_dir, model_filename)
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # 生成并保存混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(1).cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    # 获取类别标签
    class_names = [f'Class {i}' for i in range(Config.num_classes)]
    # 调用新的混淆矩阵绘制函数
    plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')

    # 保存训练配置
    with open(os.path.join(results_dir, 'training_config.txt'), 'w') as f:
        for key, value in vars(Config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    main()