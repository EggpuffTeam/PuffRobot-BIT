# -*- coding: utf-8 -*-
from selfclass.yolo11 import SoloDetection

# 使用范例
# 定义类别标签
classes = ["red", "yellow", "green"]

# 创建推理对象
detector = SoloDetection("models/yolov11s/yolo11s-trafficlight.onnx",
                         confidence_thres=0.5,
                         iou_thres=0.5,
                         class_names=classes)

# 检测
image_path = "images/trafficlights/trafficlight.jpg"
result = detector.detect_image(image_path)

# 输出检测结果
if result:
    print(f"{result.x}, {result.y}, {result.w}, {result.h}, {result.label}, {result.score:.2f}")
    detector.visualize_on_source(result, image_path=image_path)
else:
    # 红绿灯没检测到时，默认为红灯。不需要的话可以注释掉改为另一个print
    print(f"0,0,0,0, red, 0.00")
    # print(f"未检测到")
