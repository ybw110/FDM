import pandas as pd

# 读取 CSV 文件
file_path = '/data1/ybw/3D/ybw_data_3cam/my_picture/20240823204519dataset=print5-10/data.csv'
df = pd.read_csv(file_path)

# 将 z_offset 列的所有值改为 -0.1
df['z_offset'] = -0.1

# 保存修改后的数据到原文件
df.to_csv(file_path, index=False)

print(f"文件已成功修改并保存回 '{file_path}'")