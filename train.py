from ultralytics import YOLO

from ultralytics import YOLO
 
# 从头开始创建一个新的YOLO模型
model = YOLO('yolo11n.yaml')
 
# 加载预训练的YOLO模型（推荐用于训练）
model = YOLO('yolo11n.pt')
 
# 使用数据集训练模型epochs个周期
results = model.train(data='mis.yaml', epochs=50, batch=4)
 
# 评估模型在验证集上的性能
results = model.val()
 
# 使用模型对图片进行目标检测
results = model('test.jpg')
 
# 将模型导出为ONNX格式
success = model.export(format='onnx')

for result in results:
    result.show()
