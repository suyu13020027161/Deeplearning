import os
import pandas as pd

# 设置输入文件和输出目录路径
input_csv = 'labels.csv'  # 原始CSV文件路径
output_dir = '/home/ysu/Deeplearning/linedetect/lab'  # 输出CSV文件的存储路径


# 确保输出目录存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件并命名列
data = pd.read_csv(input_csv, header=None, names=['label', 'x1', 'y1', 'x2', 'y2', 'filename', 'width', 'height'])

# 获取第六列 'filename' 中唯一的文件名
unique_filenames = data['filename'].unique()

# 按照文件名分组并保存为单独的CSV文件
for filename in unique_filenames:
    # 获取文件名，不带后缀
    file_base_name = os.path.splitext(filename)[0]
    
    # 筛选出 'filename' 列中等于当前文件名的数据
    subset_data = data[data['filename'] == filename]
    
    # 删除文件名列
    subset_data = subset_data.drop(columns=['filename'])
    
    # 生成输出CSV文件路径，不带后缀
    output_csv = os.path.join(output_dir, f"{file_base_name}.csv")
    
    # 保存数据到指定路径的CSV文件
    subset_data.to_csv(output_csv, index=False, header=False)
    print(f"Saved {output_csv}")

