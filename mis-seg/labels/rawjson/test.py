from PIL import Image
import os

def get_image_sizes(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"{filename}: 宽度={width}, 高度={height}")

# 用你的图片文件夹路径替换下面的字符串
folder_path = '/home/ysu/Deeplearning/datasets/mis-seg/images/train'
get_image_sizes(folder_path)

