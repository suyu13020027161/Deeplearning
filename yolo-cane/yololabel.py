import os
import pandas as pd
import numpy as np

def csv_to_yolo_label(input_csv_path, output_dir, width):
    df = pd.read_csv(input_csv_path, header=None,
                     names=['label', 'x1', 'y1', 'x2', 'y2', 'img_width', 'img_height'])
    
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_txt_path = os.path.join(output_dir, f"{base_filename}.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_txt_path, 'w') as file:
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            img_width, img_height = row['img_width'], row['img_height']
            # 端点坐标转换为 NumPy 数组
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])    
            # 计算方向向量
            direction = p2 - p1
            direction_length = np.linalg.norm(direction)
            # 计算单位方向向量
            unit_direction = direction / direction_length
            # 计算垂直单位向量（顺时针旋转90度）
            perp_direction = np.array([-unit_direction[1], unit_direction[0]])
            # 半宽度
            half_width = width / 2
            # 计算四个顶点
            vertex1 = p1 + perp_direction * half_width  # 端点1的一侧
            vertex2 = p1 - perp_direction * half_width  # 端点1的另一侧
            vertex3 = p2 - perp_direction * half_width  # 端点2的一侧
            vertex4 = p2 + perp_direction * half_width  # 端点2的另一侧
            x1 = vertex1[0] / img_width
            y1 = vertex1[1] / img_height
            x2 = vertex2[0] / img_width
            y2 = vertex2[1] / img_height
            x3 = vertex3[0] / img_width
            y3 = vertex3[1] / img_height            
            x4 = vertex4[0] / img_width
            y4 = vertex4[1] / img_height
            file.write(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")

def process_all_csv(input_dir, output_dir, width):
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_csv_path = os.path.join(input_dir, filename)
            csv_to_yolo_label(input_csv_path, output_dir, width)

# 设置输入和输出目录
input_dir = '/home/ysu/Deeplearning/linedetect/YOLO based/lab'  # 输入 CSV 文件夹路径
output_dir = '/home/ysu/Deeplearning/linedetect/YOLO based/yololab'  # 输出标签文件存放路径

width = 50
process_all_csv(input_dir, output_dir, width)

