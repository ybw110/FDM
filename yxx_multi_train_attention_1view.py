import csv
import datetime
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from yxx_multi_model_attention_1view import MultiViewNet
from ybw_single_model import *
from ybw_dataloader import get_data_loaders, set_seed
import ybw_config as Config
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', colour='RED', ncols=120)

    for inputs, labels in loop:
        # 打印输入的维度，确保它是 4D 的
        # print(f"Input shape before model: {inputs.shape}")
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs,_ = model(inputs)
        # print(f"Output shape after model: {outputs.shape}")  # 打印输出形状
        loss = criterion(outputs, labels.argmax(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.argmax(1)).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_accuracy = 100. * correct / total
    return avg_loss, avg_accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs,_ = model(inputs)
            loss = criterion(outputs, labels.argmax(1))

            total_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.argmax(1)).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )

    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels

def plot_losses_and_accuracies(train_losses, val_losses, train_accuracies, val_accuracies, epochs, results_dir):
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    colors = {
        'train_loss': '#F98F34', 'val_loss': '#FFBC80',
        'train_acc': '#0C4E9B', 'val_acc': '#6B98C4',
    }

    ax1.plot(range(epochs), train_losses, color=colors['train_loss'], linestyle='-', linewidth=2, label='Train Loss')
    ax1.plot(range(epochs), val_losses, color=colors['val_loss'], linestyle='--', linewidth=2, label='Val Loss')
    ax2.plot(range(epochs), train_accuracies, color=colors['train_acc'], linestyle='-', linewidth=2, label='Train Accuracy')
    ax2.plot(range(epochs), val_accuracies, color=colors['val_acc'], linestyle='--', linewidth=2, label='Val Accuracy')

    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.title('Loss and Accuracy', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_accuracy_plot.png'), dpi=300)
    plt.close()

def plot_metric(metric_values, metric_name, epochs, color, results_dir):
    plt.figure(figsize=(12, 6.75))
    plt.plot(range(epochs), metric_values, color=color, linestyle='-', linewidth=4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().tick_params(width=2, length=6, labelsize=12)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.title(f'{metric_name} over Epochs', fontsize=18)
    plt.grid(True, linestyle=':', alpha=0.5, which='both', color='lightgray')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{metric_name.lower()}_plot.png'), dpi=300)
    plt.close()

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

def plot_class_accuracies(all_labels, all_predictions, class_names, results_dir):
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    for i in range(len(all_labels)):
        label = all_labels[i]
        pred = all_predictions[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    class_accuracies = [100 * correct / total for correct, total in zip(class_correct, class_total)]

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accuracies)
    plt.title('Accuracy by Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(results_dir, 'class_accuracies.png'))
    plt.close()

def save_metrics_to_csv(results_dir, train_losses, val_losses, train_accs, val_accs, precisions, recalls, f1_scores):
    file_path = os.path.join(results_dir, 'training_metrics.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy', 'Precision', 'Recall', 'F1-Score'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1, train_losses[epoch], val_losses[epoch], train_accs[epoch],
                val_accs[epoch], precisions[epoch], recalls[epoch], f1_scores[epoch]
            ])

# 创建基于日期和时间的结果目录
# 选择对应的前缀
author_name = 'single_view'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 将作者名字作为前缀加入文件夹命名
results_dir = os.path.join('multi_results', f"{author_name}_{current_time}")
os.makedirs(results_dir, exist_ok=True)

def main():
    set_seed(Config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    with open(os.path.join(results_dir, 'config.txt'), 'w') as f:
        for key, value in vars(Config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")

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

    model = MultiViewNet(num_classes=Config.num_classes).to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=Config.factor, patience=Config.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    precisions, recalls, f1_scores = [], [], []

    metric_colors = {
        'precision': '#FF9A9B',
        'recall': '#9998FF',
        'f1': '#C99BFF',
    }

    for epoch in range(Config.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                                                     Config.epochs)
        val_loss, val_accuracy, precision, recall, f1, val_preds, val_labels = validate_model(model, val_loader,
                                                                                              criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"Epoch {epoch + 1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        scheduler.step(val_loss)

        is_best = val_accuracy > best_val_acc
        best_val_acc = max(val_accuracy, best_val_acc)
        model_filename = f"epoch={epoch + 1:02d}-val_loss={val_loss:.2f}-val_acc={val_accuracy:.2f}.pth"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join(results_dir, model_filename))

    save_metrics_to_csv(results_dir, train_losses, val_losses, train_accs, val_accs, precisions, recalls, f1_scores)
    plot_losses_and_accuracies(train_losses, val_losses, train_accs, val_accs, Config.epochs, results_dir)
    plot_metric(precisions, 'Precision', Config.epochs, metric_colors['precision'], results_dir)
    plot_metric(recalls, 'Recall', Config.epochs, metric_colors['recall'], results_dir)
    plot_metric(f1_scores, 'F1 Score', Config.epochs, metric_colors['f1'], results_dir)

    class_names = ['Severe Under-extrusion', 'Mild Under-extrusion', 'Normal', 'Mild Over-extrusion',
                   'Severe Over-extrusion']
    report = classification_report(val_labels, val_preds, target_names=class_names, digits=5)
    print(report)

    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    plot_class_accuracies(val_labels, val_preds, class_names, results_dir)


if __name__ == '__main__':
    main()