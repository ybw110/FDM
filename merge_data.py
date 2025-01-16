import os
import shutil
import pandas as pd
import glob


# 定义源文件夹和目标文件夹
base_path = '/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed'

# 获取所有以 "dataset=print" 开头的文件夹
source_folders = [f for f in os.listdir(base_path) if f.startswith('20') and 'dataset_else=print' in f]
# source_folders = [f for f in os.listdir(base_path) if f.startswith('20') and 'dataset_else=print' in f and 'print-radom' not in f]
# 打印总文件夹数
print(f"总共找到 {len(source_folders)} 个源文件夹")
# 设置目标文件夹路径
target_folder = os.path.join(base_path, 'merge_data_all')

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)
os.makedirs(os.path.join(target_folder, '45'), exist_ok=True)
os.makedirs(os.path.join(target_folder, '90'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'besides'), exist_ok=True)

# 初始化一个空的DataFrame来存储所有CSV数据
all_csv_data = pd.DataFrame()

# 初始化图片计数器
image_counter = 1

# 遍历每个源文件夹
for folder in source_folders:
    full_folder_path = os.path.join(base_path, folder)
    print(f"\n处理文件夹: {folder}")
    # 读取CSV文件
    csv_file = os.path.join(full_folder_path, 'data.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"  CSV文件行数: {len(df)}")

        # 创建字典来存储旧文件名到新文件名的映射
        name_mapping = {}

        # 处理每个角度的图片
        for angle in ['45', '90', 'besides']:
            source_path = os.path.join(full_folder_path, f'{folder}-{angle}')
            target_path = os.path.join(target_folder, angle)

            # 获取该角度下的所有图片并按原始顺序排序
            images = sorted(glob.glob(os.path.join(source_path, '*.jpg')))
            print(f"  {angle}度角图片数量: {len(images)}")

            for img in images:
                old_img_name = os.path.basename(img)
                old_img_base = os.path.splitext(old_img_name)[0]

                # 如果这个基础名称还没有新的映射，创建一个
                if old_img_base not in name_mapping:
                    new_img_base = f'image-{image_counter:04d}'
                    name_mapping[old_img_base] = new_img_base
                    image_counter += 1

                new_img_name = f'{name_mapping[old_img_base]}.jpg'

                # 复制图片到新位置
                shutil.copy(img, os.path.join(target_path, new_img_name))

        # 更新CSV数据中的img_path
        df['img_path'] = df['img_path'].map(name_mapping)
        # 处理img_path中的NaN值，可以选择删除这些行，或者使用默认值替换
        df = df.dropna(subset=['img_path'])  # 删除img_path为NaN的行
        # 提取数字并转换为整数
        df['img_num'] = df['img_path'].str.extract('(\d+)').astype(int)

        # 添加处理后的数据到总DataFrame
        all_csv_data = pd.concat([all_csv_data, df], ignore_index=True)

# 保存合并后的CSV文件
all_csv_data.to_csv(os.path.join(target_folder, 'merged_data.csv'), index=False)

print(f"合并完成! 总共处理了 {image_counter - 1} 组图片（每组包含3个角度）。")