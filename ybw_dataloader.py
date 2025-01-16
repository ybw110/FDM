from collections import Counter

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import transforms
from tqdm import tqdm

from ybw_dataset import ParametersDataset, set_seed
from sklearn.model_selection import train_test_split
import torch

def get_transforms(train=True, angle=45):
    # 根据角度选择相应的均值和标准差,5组数据集的
    # merge - data - radom:
    # if angle == 45:
    #     MEAN = [0.3452, 0.3804, 0.4261]
    #     STD = [0.2656, 0.2769, 0.3000]
    # elif angle == 90:
    #     MEAN = [0.4124, 0.4363, 0.4868]
    #     STD = [0.2735, 0.2696, 0.2823]
    # else:  # besides
    #     MEAN = [0.4002, 0.4100, 0.3871]
    #     STD = [0.2855, 0.2829, 0.2926]

    # merge - data - fixed:
    # if angle == 45:
    #     MEAN = [0.3148, 0.3729, 0.4246]
    #     STD = [0.2727, 0.2826, 0.3225]
    # elif angle == 90:
    #     MEAN = [0.3067, 0.3332, 0.3712]
    #     STD = [0.2338, 0.2435, 0.2837]
    # else:  # besides
    #     MEAN = [0.3332, 0.3531, 0.3221]
    #     STD = [0.2707, 0.2714, 0.2839]

    # merge - data - all:
    if angle == 45:
        MEAN = [0.3260, 0.3757, 0.4252]
        STD = [0.2702, 0.2806, 0.3142]
    elif angle == 90:
        MEAN = [0.3457, 0.3713, 0.4139]
        STD = [0.2488, 0.2535, 0.2835]
    else:  # besides
        MEAN = [0.3579, 0.3742, 0.3461]
        STD = [0.2763, 0.2757, 0.2873]

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])




def get_data_loaders(csv_file, root_dir, angle, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_workers=4):
    # 创建完整数据集
    full_dataset = ParametersDataset(csv_file, root_dir, angle, transform=get_transforms(train=True, angle=angle))
    # 计算每个集合的大小
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_multi_data_loaders(csv_file, root_dir, angles, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,num_workers=4):
    datasets = []
    for angle in angles:
        dataset = ParametersDataset(csv_file, root_dir, angle, transform=get_transforms(train=True, angle=angle))
        datasets.append(dataset)

    combined_dataset = MultiViewDataset(datasets)

    dataset_size = len(combined_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


class MultiViewDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert all(len(dataset) == len(self.datasets[0]) for dataset in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx][0] for dataset in self.datasets], self.datasets[0][idx][1]


def count_labels(dataset):
    """Helper function to count the number of samples in each class for a given dataset, with a progress bar."""
    label_counter = Counter()

    for data, labels in tqdm(dataset, desc="Counting labels", leave=True):
        # Assuming labels are one-hot encoded
        if isinstance(labels, torch.Tensor):
            if labels.ndim > 1:
                # For batched data
                for label in labels:
                    label_index = torch.argmax(label).item()
                    label_counter[label_index] += 1
            else:
                # For single sample
                label_index = torch.argmax(labels).item()
                label_counter[label_index] += 1
        else:
            # If labels are not tensors (unlikely in this case, but just for completeness)
            label_counter[labels] += 1

    return label_counter


def print_label_distribution(dataset_name, label_counter):
    """Helper function to print the distribution of labels."""
    print(f"\n{dataset_name} label distribution:")
    for label, count in label_counter.items():
        print(f"Label {label}: {count} samples")


def main():
    csv_file = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_fixed/data_analysis.csv"
    root_dir = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_fixed"
    angles = ["45", "90", "besides"]

    # print("Testing single view data loader:")
    # for angle in angles:
    #     print(f"\nTesting angle: {angle}")
    #     train_loader, val_loader, test_loader = get_data_loaders(csv_file, root_dir, angle)
    #
    #     # 打印数据集大小
    #     print(f"Train set size: {len(train_loader.dataset)}")
    #     print(f"Validation set size: {len(val_loader.dataset)}")
    #     print(f"Test set size: {len(test_loader.dataset)}")
    #
    #     # 检查一个批次的数据
    #     for images, labels in train_loader:
    #         print(f"Batch shape: {images.shape}")
    #         print(f"Labels shape: {labels.shape}")
    #         print(f"Sample image range: ({images.min().item()}, {images.max().item()})")
    #         print(f"Sample label: {labels[0]}")
    #         break

    print("\nTesting multi-view data loader:")
    multi_train_loader, multi_val_loader, multi_test_loader = get_multi_data_loaders(csv_file, root_dir, angles)

    # 统计并打印多视角训练集、验证集和测试集的类别分布
    multi_train_labels = count_labels(multi_train_loader.dataset)
    multi_val_labels = count_labels(multi_val_loader.dataset)
    multi_test_labels = count_labels(multi_test_loader.dataset)

    print_label_distribution("Multi-view Train", multi_train_labels)
    print_label_distribution("Multi-view Validation", multi_val_labels)
    print_label_distribution("Multi-view Test", multi_test_labels)

    # 打印多视角数据集大小
    print(f"Multi-view Train set size: {len(multi_train_loader.dataset)}")
    print(f"Multi-view Validation set size: {len(multi_val_loader.dataset)}")
    print(f"Multi-view Test set size: {len(multi_test_loader.dataset)}")

    # 检查一个批次的多视角数据
    for images, labels in multi_train_loader:
        print(f"Number of views: {len(images)}")
        for i, view_images in enumerate(images):
            print(f"View {i+1} batch shape: {view_images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample label: {labels[0]}")
        break

    # 检查验证集的一个批次
    for images, labels in multi_val_loader:
        print(f"Multi-view Validation batch shapes: {[img.shape for img in images]}")
        break

    # 检查测试集的一个批次
    for images, labels in multi_test_loader:
        print(f"Multi-view Test batch shapes: {[img.shape for img in images]}")
        break

if __name__ == '__main__':
    set_seed(42)
    main()


