#苏雨的json数据转换为txt程序
import os
import json
from PIL import Image

#用你的图片文件夹路径替换下面的字符串（苏雨）
folder_path = '/home/ysu/Deeplearning/datasets/mis-seg/images/val'

# Define the class labels 
class_labels = {"car": 0, "bike": 1, "plane": 2} # Change/add more for your database

# Define the directories
input_dir = '/home/ysu/Deeplearning/datasets/mis-seg/labels/rawjson' # Replace with your directory
output_dir = '/home/ysu/Deeplearning/datasets/mis-seg/labels/rawjson' # Replace with your directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#定义名称数据库（苏雨）
namelist = []
#定义长数据库（苏雨）
wlist = []
#定义宽数据库（苏雨）
hlist = []


def get_image_sizes(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                namelist.append(filename)
                wlist.append(width)
                hlist.append(height)
                #print(f"{filename}: 宽度={width}, 高度={height}")
get_image_sizes(folder_path)





for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        #寻找对应图片尺寸信息（苏雨）
        i = 0
        while i < len(namelist):
            #分割文件扩展名（苏雨）
            base_name1, extension1 = os.path.splitext(filename)
            base_name2, extension2 = os.path.splitext(namelist[i])
            if base_name1 == base_name2:
                num = i
                with open(os.path.join(input_dir, filename)) as f:
                    data = json.load(f)
                with open(os.path.join(output_dir, filename.replace('.json', '.txt')), 'w') as out_file: 
                    for shape in data.get('shapes', []): 
                        out_file.write(f"{0}")
                        points = shape.get('points', [])
                        for point in points:                
                            #需要归一化（苏雨）
                            x = point[0] / wlist[num]
                            y = point[1] / hlist[num]
                            out_file.write(f" {x} {y}")
                        out_file.write(f"\n")                              
                break
            else:
                i = i + 1
        
       

        
            






        

 
            

 
 
 
 
 
 
 
 
 
 
 
 
 
 
            

                    


               

print("Conversion completed successfully!")
