#苏雨的关键点预测结果读取并显示程序
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

image_path = '/home/ysu/Deeplearning/linedetect/YOLO based/1.jpg'
model = YOLO("/home/ysu/Deeplearning/linedetect/YOLO based/pick500.pt")
conf = 0.5 



results = model.predict(image_path, conf=conf, show=False, show_boxes=False, save_txt=False)  # return a list of Results objects

image = cv2.imread(image_path)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen

    
numpy_array = (keypoints.xy).numpy()

#print (numpy_array)
# Initialize four empty lists to store the data from each column
x1 = []
y1 = []
x2 = []
y2 = []

i = 0
while i<len(numpy_array):
    x1.append(numpy_array[i][0][0])
    y1.append(numpy_array[i][0][1])
    x2.append(numpy_array[i][1][0])
    y2.append(numpy_array[i][1][1])
    i = i + 1
    
i = 0    
while i < len(x1):
    x1_abs = int(x1[i])
    y1_abs = int(y1[i])
    x2_abs = int(x2[i])
    y2_abs = int(y2[i])
    point1 = (x1_abs, y1_abs)
    point2 = (x2_abs, y2_abs)    
    cv2.circle(image, (x1_abs, y1_abs), radius=5, color=(0, 0, 255), thickness=20)
    cv2.circle(image, (x2_abs, y2_abs), radius=5, color=(0, 255, 0), thickness=20)
    cv2.line(image, point1, point2, color=(0, 255, 255), thickness=10)
    i = i+1

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Image with canes marked')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

