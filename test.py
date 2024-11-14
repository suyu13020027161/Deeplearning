import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.backend import mean, sum

# 设置图像和标注的尺寸
IMG_WIDTH = 600
IMG_HEIGHT = 800

def dice_loss(y_true, y_pred):
    numerator = 2 * sum(y_true * y_pred)
    denominator = sum(y_true + y_pred)
    return 1 - numerator / denominator


def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []

    # 列出所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.jpg')  
        
        # 读取图片和标注
        img = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        
        if img is not None and label is not None:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT))

            #plt.imshow(label)
            #plt.axis('off')
            #plt.show()            
            
            
            images.append(img)
            labels.append(label)



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

def build_model():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    image_path = '/home/ysu/Deeplearning/linedetect/img'  # 图像文件夹路径
    label_path = '/home/ysu/Deeplearning/linedetect/labimg'  # 标注文件夹路径

    images, labels = load_images_and_labels(image_path, label_path)
    images = images.astype('float32') / 255
    labels = labels.astype('float32') / 255

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(train_images, train_labels, epochs=2, batch_size=2, validation_split=0.1)
    model.save('fig.h5')



if __name__ == '__main__':
    main()

