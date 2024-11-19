import cv2
import numpy as np

import cv2
import numpy as np

def resize_image(image, scale=None, max_width=800, max_height=600):
    """根据最大尺寸或缩放比例调整图像大小"""
    height, width = image.shape[:2]
    
    # 如果提供了缩放比例，则按比例缩放
    if scale:
        width = int(width * scale)
        height = int(height * scale)
    else:
        # 否则根据最大宽高进行缩放
        if width > max_width or height > max_height:
            # 计算缩放比例
            ratio = min(max_width / width, max_height / height)
            width = int(width * ratio)
            height = int(height * ratio)

    # 调整图像大小
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def draw_polygon_labels(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    image = resize_image(image)  # 调整图像大小

    height, width = image.shape[:2]

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = np.array([[(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]])
        points = np.array([(int(x * width), int(y * height)) for x, y in points.reshape(-1, 2)], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        #cv2.putText(image, str(class_id), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Image with Polygon Labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例用法
image_path = 'IMG20241018142140.jpg'
label_path = 'IMG20241018142140.txt'
draw_polygon_labels(image_path, label_path)

