#苏雨的图像预处理程序
import cv2
import numpy as np

#首先二值化处理图像（苏雨）
def binary_thresholding_with_size_limit(image_path, max_width, max_height, threshold=127, max_value=255):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接以灰度图形式读取
    if image is None:
        print("Error: Image could not be read.")
        return

    # 获取原始图像尺寸
    height, width = image.shape[:2]

    # 计算缩放比例
    scale_width = max_width / width if width > max_width else 1
    scale_height = max_height / height if height > max_height else 1
    scale = min(scale_width, scale_height)  # 选择较小的比例以保持纵横比

    # 缩放图像
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # 应用阈值二值化
    _, binary_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)

    # 显示结果
    #cv2.imshow('Binary Image', binary_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 可选：保存二值化后的图像
    #cv2.imwrite('binary_image.jpg', binary_image)
    return (binary_image)


#其次对图像降噪（苏雨）
def remove_noise(image, kernel_high, kernel_len):
    #第一个是高，第二个是宽（苏雨）
    kernel_size=(kernel_high, kernel_len)
    # 创建形态学操作的核
    kernel = np.ones(kernel_size, np.uint8)

    # 开运算去除噪点
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 显示原始图像和处理后的图像
    #cv2.imshow('Original Image', image)
    #cv2.imshow('Noise Removed Image', opening)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 可选：保存去噪后的图像
    #cv2.imwrite('noise_removed_image.jpg', opening)
    return (opening)






def retain_specific_shapes_by_length(image, min_length):


    # 确保图像是二值化的
    if not np.array_equal(np.unique(image), [0, 255]):
        print("Warning: Image is not binary, applying binary threshold.")
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑背景的图像
    retained_shapes = np.zeros_like(image)

    # 检查每个轮廓的长度，并保留符合长度要求的形状
    for contour in contours:
        length = cv2.arcLength(contour, closed=True)  # 可以设定为True或False根据形状是否闭合
        if length >= min_length:
            cv2.drawContours(retained_shapes, [contour], -1, (255), thickness=cv2.FILLED)  # 使用填充来保留形状的原始外观

    # 显示和保存处理后的图像
    #cv2.imshow('Retained Shapes', retained_shapes)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 可选：保存结果图像
    #cv2.imwrite('retained_shapes.jpg', retained_shapes)
    return (retained_shapes)    


#定义一些必要的调整参数（苏雨）
img_path = 'IMG20241018142140.jpg'
out_img_len = 600
out_img_high = 800
kernel_high = 5
kernel_len = 5
min_length = 50

#调用函数（苏雨）
binary_image = binary_thresholding_with_size_limit(img_path, out_img_len, out_img_high)
opening = remove_noise(binary_image, kernel_high, kernel_len)
retained_shapes = retain_specific_shapes_by_length(opening, min_length)
cv2.imshow('binary_image', binary_image)
cv2.imshow('opening', opening)
cv2.imshow('Retained Shapes', retained_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
