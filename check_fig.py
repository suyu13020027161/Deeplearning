import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    img /= 255.0  # 归一化到0-1
    return img

def load_label(label_path):
    label_img = load_img(label_path, color_mode="grayscale", target_size=(800, 600))
    label_img = img_to_array(label_img)
    label_img = np.squeeze(label_img)  # 移除不必要的维度
    label_img = label_img.astype(np.uint8)  # 确保标签是整数类型
    return label_img

model = load_model('fig.h5')

def predict_image(model, image_path):
    img = preprocess_image(image_path, target_size=(800, 600))  # 指定图片尺寸
    prediction = model.predict(img)
    return prediction[0]  # 返回预测结果

def compare_predictions(predictions, labels):
    # 阈值处理预测结果以得到二值图像
    predictions = (predictions > 0.5).astype(np.uint8)

    # 为了确保数据维度正确，使用squeeze去掉大小为1的维度
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)

    # 计算准确率和其他统计数据
    acc = accuracy_score(labels.flatten(), predictions.flatten())
    report = classification_report(labels.flatten(), predictions.flatten())
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(predictions, cmap='gray')
    plt.title('Predicted Label')
    plt.subplot(1, 3, 2)
    plt.imshow(labels, cmap='gray')
    plt.title('True Label')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(predictions - labels), cmap='gray')
    plt.title('Difference')
    plt.show()


# 设定图片和标签路径
image_path = '1.jpg'
label_path = '1l.jpg'

# 进行预测
predictions = predict_image(model, image_path)

# 加载标签
labels = load_label(label_path)

# 比较并打印结果
compare_predictions(predictions, labels)
