#这个的loss是有区别于原本论文的
"""根据论文的描述,这两种损失计算方式确实有所不同:

原始论文的方法:
对每个视角v单独计算损失Lv,然后进行加权求和:
loss = Σ (αv^γ * Lv)
其中αv是每个视角的权重,γ是一个大于1的指数参数。

修改后的方法:
先对各视角的输出进行加权平均:
output_weighted = Σ (αv * outputv)
然后基于这个加权平均的输出计算一次损失。

分析这两种方法:
原始方法保留了每个视角的独立性,可以更好地捕捉各视角的特征。通过加权求和,可以灵活调整各视角的重要性。
修改后的方法简化了计算,可能计算效率更高。但可能会丢失一些视角间的差异信息。
原始方法可以更精细地控制各视角的贡献,特别是通过γ参数的调节。
修改后的方法对各视角的融合更直接,可能有利于学习视角间的互补性。"""
import datetime
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from yxx_multi_model_all import MultiViewNet
from ybw_dataloader import get_data_loaders, set_seed,get_multi_data_loaders
import ybw_config as Config
import seaborn as sns


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs, gamma):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', colour='RED', ncols=120)

    weight_var = torch.ones(Config.num_views).to(device) * (1/Config.num_views)

    for inputs, labels in loop:
        # 重置 weight_var 的维度
        weight_var = weight_var.view(-1)  # 确保 weight_var 是一维的
        inputs = [input.to(device) for input in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        Output_list, _  = model(inputs)

        # 计算加权平均的输出
        output_var = torch.stack(Output_list)
        weight_var = weight_var.unsqueeze(1).unsqueeze(2).expand(-1, labels.size(0), Config.num_classes)
        output_weighted = (weight_var * output_var).sum(0)

        # 计算总的损失
        loss = criterion(output_weighted, labels.argmax(1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        _, predicted = torch.max(output_weighted.data, 1)
        correct += (predicted == labels.argmax(1)).sum().item()

        # 更新视角权重
        with torch.no_grad():
            weight_up_list = []
            for v in range(len(Output_list)):
                loss_temp = criterion(Output_list[v], labels.argmax(1))
                weight_up_temp = loss_temp ** (1/(1-gamma))
                weight_up_list.append(weight_up_temp)
            weight_up_var = torch.stack(weight_up_list)
            weight_down_var = weight_up_var.sum()
            weight_var = weight_up_var / weight_down_var

        # 更新进度条
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_accuracy = 100. * correct / total
    # 在返回之前，确保 weight_var 是一维的
    return avg_loss, avg_accuracy, weight_var.view(-1)

def validate_model(model, val_loader, criterion, device, gamma, weight_var):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)

            Output_list, _  = model(inputs)

            # 计算加权平均的输出
            output_var = torch.stack(Output_list)
            weight_var_expanded = weight_var.unsqueeze(1).unsqueeze(2).expand(-1, labels.size(0), Config.num_classes)
            output_weighted = (weight_var_expanded * output_var).sum(0)

            # 计算总的损失
            loss = criterion(output_weighted, labels.argmax(1))

            total_loss += loss.item()
            total += labels.size(0)

            _, predicted = torch.max(output_weighted.data, 1)
            correct += (predicted == labels.argmax(1)).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_predictions, all_labels


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(12, 7))

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})

    # 设置两个y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制损失曲线
    ax1.plot(range(epochs), train_losses, color='blue', linestyle='-', linewidth=2.5, label='Train Loss')
    ax1.plot(range(epochs), val_losses, color='blue', linestyle='--', linewidth=2.5, label='Val Loss')

    # 绘制准确率曲线
    ax2.plot(range(epochs), train_accuracies, color='orange', linestyle='-', linewidth=2.5, label='Train Accuracy')
    ax2.plot(range(epochs), val_accuracies, color='orange', linestyle='--', linewidth=2.5, label='Val Accuracy')

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

    plt.rcParams.update({'font.size': 12})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True)

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# 创建保存结果的目录
# 选择对应的前缀
author_name = 'multi'
# 获取当前时间并格式化
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 将作者名字作为前缀加入文件夹命名
results_dir = os.path.join('multi_results', f"{author_name}_{current_time}")
os.makedirs(results_dir, exist_ok=True)

def main():
    set_seed(Config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 保存配置信息
    with open(os.path.join(results_dir, 'config.txt'), 'w') as f:
        for key, value in vars(Config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")

    train_loader, val_loader, test_loader = get_multi_data_loaders(
        Config.csv_file, Config.root_dir, Config.angles,
        batch_size=Config.batch_size,
        train_ratio=Config.train_ratio,
        val_ratio=Config.val_ratio,
        test_ratio=Config.test_ratio,
        num_workers=Config.num_workers
    )

    model = MultiViewNet(num_classes=Config.num_classes, num_views=Config.num_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=Config.factor, patience=Config.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()

    weight_var = torch.ones(Config.num_views).to(device) * (1/Config.num_views)
    gamma = Config.gamma
    best_val_acc = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(Config.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        train_loss, train_accuracy, weight_var = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, Config.epochs, gamma)
        val_loss, val_accuracy, val_preds, val_labels = validate_model(model, val_loader, criterion, device, gamma, weight_var)

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"View Weights: {weight_var.tolist()}")

        scheduler.step(val_loss)

        # 保存最佳模型
        is_best = val_accuracy > best_val_acc
        best_val_acc = max(val_accuracy, best_val_acc)
        model_filename = f"epoch={epoch+1:02d}-val_loss={val_loss:.2f}-val_acc={val_accuracy:.2f}.pth"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'weight_var': weight_var,
        }, is_best, os.path.join(results_dir, model_filename))

        # 每5个epoch绘制一次混淆矩阵
        if (epoch + 1) % 5 == 0 or epoch == Config.epochs - 1:
            cm = confusion_matrix(val_labels, val_preds)
            class_names = [f'Class {i}' for i in range(Config.num_classes)]
            plot_confusion_matrix(cm, classes=class_names, title=f'Confusion Matrix - Epoch {epoch + 1}')

        # 绘制训练过程中的指标
    plot_metrics(train_losses, val_losses, train_accs, val_accs, Config.epochs)

    # 测试最佳模型
    best_model_path = os.path.join(results_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_val_acc']:.2f}%")
    else:
        print(f"No best model found at {best_model_path}. Using the model from the last epoch.")

    test_loss, test_accuracy, test_preds, test_labels = validate_model(model, test_loader, criterion, device, gamma, weight_var)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # 生成分类报告
    class_report = classification_report(test_labels, test_preds, target_names=[f'Class {i}' for i in range(Config.num_classes)])
    print("Classification Report:")
    print(class_report)
    with open(os.path.join(results_dir, 'test_classification_report.txt'), 'w') as f:
        f.write(class_report)

    # 绘制最终的混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    class_names = [f'Class {i}' for i in range(Config.num_classes)]
    plot_confusion_matrix(cm, classes=class_names, title='Final Test Confusion Matrix')

if __name__ == '__main__':
    main()