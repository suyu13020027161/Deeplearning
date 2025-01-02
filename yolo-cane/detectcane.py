from ultralytics import YOLO
import random
import cv2
import numpy as np
 
import cv2

def resize_and_show(img):

    if img is None:
        print("Image not found")
        return

    # 获取图像原始尺寸
    height, width = img.shape[:2]

    # 计算新尺寸
    new_width = int(600)
    new_height = int(800)

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 显示图像
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



 
 
 
model = YOLO("best1.2k.pt")
img = cv2.imread("5.jpg")
 
 
# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
 
 
conf = 0.1
 
 
results = model.predict(img, conf=conf)
colors = [(0, 255, 0)]
#print(results)


for result in results:
    i = 0
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        #green,blue,red（苏雨）
        cv2.polylines(img, points, True, (0, 0, 255), 8)
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(img, points, colors[color_number])
        i = i + 1
    print('Predicting the number of canes = ', i)
 
 
 
 
resize_and_show(img)

cv2.waitKey(0)
 
 
#cv2.imwrite("/home/ysu/Deeplearning/seg/result2.jpg", img3
