import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# 加载训练好的模型
model = load_model('new.h5')
print("Model loaded successfully.")

# 获取模型的输入尺寸
model_input_shape = (model.input_shape[2], model.input_shape[1])  # (width, height)
print(model_input_shape)


# 预处理图像：调整大小并归一化
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)





    return image_array

# 检测线段并绘制到原始图像上
def detect_and_draw_lines(image_path, target_size):
    # 预处理图像
    input_img = preprocess_image(image_path, target_size)
    
    # 使用模型进行预测
    prediction = model.predict(input_img)

    
    
    
    # 将预测结果恢复为二维图像
    predicted_mask = (prediction[0].reshape(target_size[::-1]) * 255).astype(np.uint8)  # 恢复为(高度, 宽度)并转换为0-255
    # 对原始图像进行调整以匹配模型输入尺寸
    original_img = cv2.imread(image_path)
    original_img_resized = cv2.resize(original_img, target_size)
    #plt.imshow(predicted_mask)
    #plt.show()
    
    
    
    # 找到线段轮廓并绘制
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = original_img_resized.copy()
   
    
    for contour in contours:
        # 使用近似多边形方法找到线段
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 绘制线段
        if len(approx) > 1:
            for i in range(len(approx) - 1):
                cv2.line(annotated_img, tuple(approx[i][0]), tuple(approx[i + 1][0]), (0, 255, 0), 2)
    
    # 显示原图、预测掩码和带预测线段的图像
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Line Mask")
    plt.imshow(predicted_mask, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Annotated Image with Lines")
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    
    plt.show()

# 示例：识别并绘制图像中的线段
image_path = '1.jpg'  # 替换为实际图片路径
detect_and_draw_lines(image_path, model_input_shape)












