# -*- coding: utf-8 -*-
import os
from pathlib import Path
from selfclass.yolo11 import MultipleDetection

# ===== 模型与类别 =====
classes = ["community", "uncommunity"]
detector = MultipleDetection(
    "models/yolov11s/yolo11s-community.onnx",
    confidence_thres=0.5,
    iou_thres=0.5,
    class_names=classes
)

# ===== 要遍历的图片目录 =====
image_dir = Path("images/community")

# ===== 统计累积结果 =====
total_people_all = 0
community_count_all = 0
uncommunity_count_all = 0

# ===== 遍历目录中的所有图片 =====
for image_path in image_dir.glob("*.*"):  # 匹配所有文件
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:  # 过滤非图片
        continue

    # 检测
    results = detector.detect_all(str(image_path))

    # 当前图片统计
    total_people = len(results)
    community_count = sum(1 for r in results if r.label == "community")
    uncommunity_count = sum(1 for r in results if r.label == "uncommunity")

    # 累计
    total_people_all += total_people
    community_count_all += community_count
    uncommunity_count_all += uncommunity_count

    # 输出当前图片结果
    print(f"[{image_path.name}] 共 {total_people} 人，其中社区人员 {community_count} 人，非社区人员 {uncommunity_count} 人")

    # 绘制检测结果并保存到 ./output/
    detector.visualize_on_source(results, image_path=str(image_path))

# ===== 累积统计 =====
print("=" * 60)
print(f"累计统计：共 {total_people_all} 人，其中社区人员 {community_count_all} 人，非社区人员 {uncommunity_count_all} 人")
print("=" * 60)
