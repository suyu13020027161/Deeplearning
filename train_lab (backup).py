import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os



#加载图像并创建对应的标注图像（苏雨）
def prepare_images_and_labels(image_path, csv_path):
    images = []
    labels = []
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    
    
    #获取图像文件夹中的所有文件（苏雨）
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = str(image_path)
        image_file = str(image_file)
        img_path = image_path + '/' + image_file
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            #调整图像大小（苏雨）
            img = cv2.resize(img, (600, 800))  
            images.append(img)
        #获取图像文件的基本名（不带扩展名）（苏雨）
        base_name = os.path.splitext(image_file)[0]        
        #在CSV文件夹中查找与图像文件同名的CSV文件（苏雨）
        csv_file_path = os.path.join(csv_path, f"{base_name}.csv")            
        # 读取CSV文件并显示数据（苏雨）
        data = pd.read_csv(csv_file_path, header=None, names=['label', 'x1', 'y1', 'x2', 'y2', 'img_width', 'img_height'])
        #缩放比例尺（苏雨）
        rate = 5            
        #数据存入数组（苏雨）
        #千万要记得清数组！（苏雨）
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for value in data.x1:
            x1_list.append(value/rate)
        for value in data.y1:
            y1_list.append(value/rate)                        
        for value in data.x2:
            x2_list.append(value/rate)            
        for value in data.y2:
            y2_list.append(value/rate)            
                                
        mask = create_mask_image(img, x1_list, y1_list, x2_list, y2_list, img.shape)
        #print(mask.shape)
        labels.append(mask)
        #plt.imshow(mask)
        #plt.show()
    #plt.imshow(images[5])
    #plt.axis('off')
    #plt.show()


    #数据集检测（苏雨）
    '''
    i = 0
    while i < len(labels):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("images")
        plt.imshow(images[i])

        plt.subplot(1, 2, 2)
        plt.title("labels")
        plt.imshow(labels[i])

        plt.show()
        i = i + 1
    '''
 

            
    return np.array(images), np.array(labels)

# 创建线段标注图像
def create_mask_image(img, x1_list, y1_list, x2_list, y2_list, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    i = 0
    while i < len(x1_list):
        x1 = int(x1_list[i])
        y1 = int(y1_list[i])
        x2 = int(x2_list[i])                
        y2 = int(y2_list[i])
        #白色线段（苏雨）
        cv2.line(mask, (x1, y1), (x2, y2), (255), 1)
        i = i + 1
    return mask










# 读取并解析数据
def load_data(file_path):
    data = pd.read_csv(file_path, header=None, names=['class', 'x1', 'y1', 'x2', 'y2', 'filename', 'img_width', 'img_height'])
    return data





# 构建模型
def build_model():
    model = Sequential([
        Conv2D(64, (1, 1), activation='relu', input_shape=(800, 600, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (1, 1), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (1, 1), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(600 * 800, activation='sigmoid'),
        tf.keras.layers.Reshape((800, 600))
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 主程序
annotations_path = 'labels.csv'

# 设置图像文件夹和CSV文件夹路径
image_path = '/home/ysu/Deeplearning/linedetect/img'  # 图像文件夹路径
csv_path = '/home/ysu/Deeplearning/linedetect/lab'       # CSV文件夹路径

images, masks = prepare_images_and_labels(image_path, csv_path)


train_images, test_images, train_labels, test_labels = train_test_split(images, masks, test_size=0.2, random_state=42)
train_images = np.expand_dims(train_images, axis=-1)  # 增加一个维度以匹配CNN输入
test_images = np.expand_dims(test_images, axis=-1)





model = build_model()



history = model.fit(train_images, train_labels, epochs=10, batch_size=5, validation_split=0.3)
model.save('lab.h5')

model.evaluate(test_images, test_labels)









