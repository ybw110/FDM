import datetime
import os
import re
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from yxx_multi_model_attention import MultiViewNet
# from yxx_multi_model_attention_with_visualization import MultiViewNet
from yxx_multi_model_contrast import *
from ybw_dataloader import get_multi_data_loaders
import ybw_config as Config
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from yxx_multi_model_attention_1view import MultiViewNet



def validate_model(model, val_loader, criterion, device, visualize_indices, results_dir):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_features = []  # 用于保存特征
    all_labels_for_tsne = []  # 用于保存用于 T-SNE 的标签

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Testing", leave=False)
        for batch_index, (inputs, labels) in enumerate(progress_bar):
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)

            # final_classification, ResNet_outputs, SelfAttention_outputs, ViewAttention_output = model(inputs)
            final_classification, features= model(inputs)
            loss = criterion(final_classification, labels.argmax(1))

            total_loss += loss.item()
            total += labels.size(0)

            _, predicted = torch.max(final_classification.data, 1)
            correct += (predicted == labels.argmax(1)).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(1).cpu().numpy())

            # 保存用于 T-SNE 的特征和标签
            all_features.extend(features.cpu().numpy())
            all_labels_for_tsne.extend(labels.argmax(1).cpu().numpy())

            # if batch_index in visualize_indices:
            #     for i, resnet_output in enumerate(ResNet_outputs):
            #         save_or_plot_feature(resnet_output, f"batch{batch_index}_view{i}_resnet", results_dir)
            #
            #     for i, self_attention_output in enumerate(SelfAttention_outputs):
            #         save_or_plot_feature(self_attention_output, f"batch{batch_index}_view{i}_self_attention",
            #                              results_dir)
            #
            #     save_or_plot_feature(ViewAttention_output, f"batch{batch_index}_view_attention", results_dir)

            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100. * correct / total:.2f}%"})

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_predictions, all_labels, all_features, all_labels_for_tsne


def save_or_plot_feature(feature, name, results_dir):
    feature = feature.cpu().detach().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(feature, aspect='auto', cmap='viridis')
    plt.title(f"Feature Map: {name}")
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='viridis'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 如果 cmap 为 None，则使用默认的颜色映射
    if cmap is None:
        cmap = plt.cm.Blues  # 默认使用蓝色调

    # 创建图像
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=28, pad=40)  # 修改标题字体大小和颜色
    plt.colorbar()  # 添加颜色条


    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=24)  # 修改x轴标签字体大小和颜色
    plt.yticks(tick_marks, classes, fontsize=24)  # 修改y轴标签字体大小和颜色

    # 设置文本格式，百分比显示两位小数
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.  # 阈值用于设置文本颜色

    # 在每个方格中添加文本
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=22,  # 设置文本字体大小
                 color="white" if cm[i, j] > thresh else "black")  # 根据值设置文本颜色

    # 设置标签的字体大小和颜色
    plt.ylabel('True Label', fontsize=28, labelpad=20)
    plt.xlabel('Predicted Label', fontsize=28, labelpad=20)

    plt.tight_layout()  # 自动调整布局


def plot_tsne(features, labels, class_names, results_dir,custom_colors=None):


    # 将特征转换为 NumPy 数组 (如果它是列表)
    features = np.array(features)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制 T-SNE 可视化
    plt.figure(figsize=(10, 8))

    if custom_colors is None:
        custom_colors = ['#ABC6E4', '#C39398', '#FCDABA', '#A7D2BA', '#D0CADE']  # 默认五种颜色
        # 为不同类别分配自定义颜色
    for i, class_name in enumerate(class_names):
        idx = np.where(np.array(labels) == i)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=class_name,
                    color=custom_colors[i % len(custom_colors)])

    plt.legend(loc='best', fontsize=20)
    plt.title("T-SNE Visualization", fontsize=28, pad=40)
    plt.xlabel("Component 1", fontsize=28, labelpad=20)
    plt.ylabel("Component 2", fontsize=28, labelpad=20)

    # 设置坐标轴数字大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    tsne_path = os.path.join(results_dir, 'tsne_visualization.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"T-SNE visualization saved to {tsne_path}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 手动输入模型路径
    model_path = input("请输入要测试的模型文件的完整路径: ").strip()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    print(f"使用模型: {os.path.basename(model_path)}")

    # 设置结果保存目录
    # 添加时间戳到结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(model_path), f'test_results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    _, _, test_loader = get_multi_data_loaders(
        Config.csv_file, Config.root_dir, Config.angles3,
        batch_size=Config.batch_size,
        train_ratio=Config.train_ratio,
        val_ratio=Config.val_ratio,
        test_ratio=Config.test_ratio,
        num_workers=Config.num_workers
    )

    # model = SharedWeightMultiViewNet(num_classes=Config.num_classes, num_views=Config.num_views).to(device)  #相同的权重
    # model = MultiViewNet(num_classes=Config.num_classes).to(device)    #1 view
    model = MultiViewNet(num_classes=Config.num_classes, num_views=2).to(device)   #2 views
    # model = MultiViewNet(num_classes=Config.num_classes, num_views=3).to(device)  # 3views

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from {model_path}")
    print(f"Model was saved at epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_val_acc']:.2f}%")

    # 指定要绘制特征图的批次索引
    visualize_indices = [1]

    print(f"Testing on {len(test_loader.dataset)} samples")
    test_loss, test_accuracy, test_preds, test_labels, all_features, all_labels_for_tsne = validate_model(
        model, test_loader, criterion, device, visualize_indices, results_dir)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}%")

    # 生成分类报告
    class_names = [f'Class {i}' for i in range(Config.num_classes)]
    report = classification_report(test_labels, test_preds, target_names=class_names,digits=5)
    print("Classification Report:")
    print(report)

    # 计算其他指标
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Save classification report
    report_path = os.path.join(results_dir, 'test_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # 绘制最终的混淆矩阵
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F8F4F2",'#1B3B70'])
    # 计算并绘制混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Test Confusion Matrix', cmap=cmap)

    cm_path = os.path.join(results_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # 绘制 T-SNE
    print("Generating T-SNE visualization...")
    plot_tsne(all_features, all_labels_for_tsne, class_names, results_dir)

if __name__ == '__main__':
    main()