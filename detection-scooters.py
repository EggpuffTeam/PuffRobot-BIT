# -*- coding: utf-8 -*-
from selfclass.yolo11 import MultipleDetection

# ===== 使用 =====
# 在这里
classes = ["stand", "fallen"]
detector = MultipleDetection(
    "models/yolov11s/yolo11s-community.onnx",
    confidence_thres=0.5,
    iou_thres=0.5,
    class_names=classes
)
image_path = "images/scooter/scooter.jpg"
results = detector.detect_all(image_path)

# 统计
total_scooter = len(results)
stand_count = sum(1 for r in results if r.label == "stand")
fallen_count = sum(1 for r in results if r.label == "fallen")

print(f"图中共有 {total_scooter} 辆电动车，其中站有 {stand_count} 辆，倒下 {fallen_count} 辆")

# 绘制所有检测结果并保存到 ./output/
detector.visualize_on_source(results, image_path=image_path)

