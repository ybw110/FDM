

# 随机的数据
# csv_file = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_radom/data_analysis.csv"
# root_dir = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_radom"

# 固定的参数
# csv_file = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_fixed/data_analysis.csv"
# root_dir = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_fixed"

# 合并的数据
csv_file = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all/data_analysis.csv"
root_dir = "/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all"

angle = "45"  # 或 "90" 或 "besides"

angles = ["45", "90", "besides"]
angles1 = ["45"]
angles2 = ["45", "90"]
angles3 = ["45", "besides"]

num_views = len(angles)
# 配置随机种子
seed = 42

# 训练配置
epochs =50
batch_size = 64
train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
num_workers=4
num_classes=5

learning_rate = 1e-3
weight_decay = 1e-5  # 正则化参数
factor=0.1        #新的学习率是旧的学习率乘以该因子。
patience=5       #在监测量多少个周期没有改善后调整学习率。

dropout_rate = 0.2


