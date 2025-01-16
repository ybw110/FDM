import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv(r'/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all/merged_data.csv',sep=',',header='infer')
df = data
# 为 DataFrame 添加了两个新列 nozzle_tip_x 和 nozzle_tip_y,我觉得有三个视角，是不是要三个
df['nozzle_tip_x_45'] = 190.0
df['nozzle_tip_y_45'] = 160.0

df['nozzle_tip_x_90'] = 380.0
df['nozzle_tip_y_90'] = 260.0

df['nozzle_tip_x_besides'] = 400.0
df['nozzle_tip_y_besides'] = 270.0

# 分类标准
category_ranges_flowrate = {
    ('flow_rate_class', 0): (20, 85.99),
    ('flow_rate_class', 1): (86, 110.99),
    ('flow_rate_class', 2): (111, 200)
}

category_ranges_feed_rate = {
    ('feed_rate_class', 0): (20, 85.99),
    ('feed_rate_class', 1): (86, 120.99),
    ('feed_rate_class', 2): (121, 200)
}

# 分类函数
def classify(value, category_ranges):
    for category, (start, end) in category_ranges.items():
        if start <= value <= end:
            return category[1]  # 返回分类标签
    return None

# 对每列数据进行分类
# 对flow_rate和feed_rate进行分类
df['flow_rate_class'] = df['flow_rate'].apply(lambda x: classify(x, category_ranges_flowrate))
df['feed_rate_class'] = df['feed_rate'].apply(lambda x: classify(x, category_ranges_feed_rate))

# 检查未分类的值
print(f"Unclassified values in flow_rate_class: {(df['flow_rate_class'].isna()).sum()}")
print(f"Unclassified values in feed_rate_class: {(df['feed_rate_class'].isna()).sum()}")

# unclassified_flow_rate = df[df['flow_rate_class'].isna()]['flow_rate']
# unclassified_feed_rate = df[df['feed_rate_class'].isna()]['feed_rate']
# print("Flow rate values that couldn't be classified:")
# print(unclassified_flow_rate)
# print("Feed rate values that couldn't be classified:")
# print(unclassified_feed_rate)

# 设置感兴趣的列
interested_columns = ['flow_rate', 'feed_rate']


# 创建一个子图,包含与 interested_columns 长度相同的行数,每行一个图像
fig, axes = plt.subplots(len(interested_columns), 1, figsize=(10, 8))

# 遍历每一列,统计该列中各元素出现的频率,并使用 frequencies.plot(kind='bar') 绘制柱状图
for i, column in enumerate(interested_columns):
    frequencies = df[column].value_counts()                     # 统计每列中各元素的出现频率
    frequencies.plot(kind='bar', ax=axes[i], color='skyblue')   # 绘制柱状图
    axes[i].set_title(f'Frequency of {column}')                 # 设置图标题和坐标轴标签
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()  # 调整子图之间的间距
plt.show()          # 显示图形

# 绘制频率直方图
fig, axes = plt.subplots(len(interested_columns), 1, figsize=(10, 8))

# 遍历每一列，绘制频率密度图
for i, column in enumerate(interested_columns):
    # 绘制频率密度图
    sns.kdeplot(df[column], ax=axes[i], color='skyblue', fill=True)
    # 设置图标题和坐标轴标签
    axes[i].set_title(f'Frequency Density of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Density')

plt.tight_layout()  # 调整子图之间的间距
plt.show()          # 显示图形



df.to_csv(r'/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all/data_analysis.csv', index=False)
df = pd.read_csv(r'/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all/data_analysis.csv', sep=',', header='infer')
#

# 指定感兴趣的列,分别统计各列中各个元素出现的次数
interested_columns = ['flow_rate_class', 'feed_rate_class']
for column in interested_columns:
    counts = df[column].value_counts()
    print(f"\nCounts for column '{column}':")
    print(counts)

# 打印最终数据行数
print(f"\nFinal data rows: {len(df)}")
