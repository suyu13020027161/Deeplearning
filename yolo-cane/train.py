from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
model = YOLO("best1.1k.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="yolobased.yaml", epochs=100, imgsz=(800, 600))
