#苏雨的关键点预测结果读取并显示程序
import cv2
import matplotlib.pyplot as plt
result_path = '/home/ysu/Deeplearning/linedetect/YOLO based/runs/pose/predict/labels/6.txt'
image_path = '/home/ysu/Deeplearning/linedetect/YOLO based/6.jpg'



# Initialize four empty lists to store the data from each column
x1 = []
y1 = []
x2 = []
y2 = []

# Open the text file and read lines
with open(result_path, 'r') as file:
    for line in file:
        # Split the line into columns based on whitespace
        columns = line.strip().split()
        
        # Check if there are enough columns
        if len(columns) >= 10:

            # Append data from each column to the respective list
            # Python uses 0-based indexing, so we subtract 1 from each column number
            x1.append(float(columns[5]))  # 6th column
            y1.append(float(columns[6]))  # 7th column
            x2.append(float(columns[8]))  # 9th column
            y2.append(float(columns[9]))  # 10th column



# Load the image

image = cv2.imread(image_path)


# Get dimensions of the image
height, width, _ = image.shape

i = 0
while i < len(x1):
    x1_abs = int(x1[i]*width)
    y1_abs = int(y1[i]*height)
    x2_abs = int(x2[i]*width)
    y2_abs = int(y2[i]*height)
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
