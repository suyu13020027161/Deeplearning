from ultralytics import YOLO
import cv2
# Load a model
model = YOLO("/home/ysu/Deeplearning/linedetect/YOLO based/yolo8.pt")  # pretrained YOLO11n model

conf = 0.1 
#results = model.predict(img, conf=conf)


# Run batched inference on a list of images
results = model.predict("6.jpg", conf=conf, show=True, show_boxes=False, save_txt=True)  # return a list of Results objects


cv2.waitKey(0)




