# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import onnxruntime as ort
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class DetectionResult:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float


class SoloDetection:
    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float, class_names: List[str]):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = class_names
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.onnx_model, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_width, self.input_height = input_shape[2], input_shape[3]

    def letterbox(self, img: np.ndarray, new_shape: tuple = (512, 512), color=(114, 114, 114)):
        """保持比例缩放并加 padding，返回缩放后的图、padding大小、缩放比例"""
        shape = img.shape[:2]  # (高,宽)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        ratio = r
        pad = (top, left)
        return img, pad, ratio

    def preprocess(self, frame: np.ndarray):
        """预处理，保存缩放比例和padding用于映射回原图"""
        self.img_height, self.img_width = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, pad, ratio = self.letterbox(img_rgb, (self.input_width, self.input_height))
        self.pad = pad
        self.ratio = ratio
        image_data = np.array(img_lb) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def detect_image(self, image_path: str) -> Optional[DetectionResult]:
        """输入图片路径，返回最佳检测结果 dataclass"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        self.source_frame = frame.copy()  # 保存原图
        img_data = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess(outputs)

    def postprocess(self, output) -> Optional[DetectionResult]:
        """解析模型输出，返回最佳检测结果并映射回原图"""
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        best_score = -1
        best_result: Optional[DetectionResult] = None

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0:4]
                # 去除 padding 并映射回原图
                x -= self.pad[1]
                y -= self.pad[0]
                left = int((x - w / 2) / self.ratio)
                top = int((y - h / 2) / self.ratio)
                width = int(w / self.ratio)
                height = int(h / self.ratio)
                if max_score > best_score:
                    best_score = max_score
                    best_result = DetectionResult(
                        x=left,
                        y=top,
                        w=width,
                        h=height,
                        label=self.classes[class_id],
                        score=max_score
                    )
        return best_result

    def visualize_on_source(self, result: DetectionResult, image_path: str):
        """
        在原图上画检测结果并保存到 ./output 文件夹下，自动适配 Windows / Linux 路径
        :param result: 检测结果
        :param image_path: 输入图片路径，用于自动命名输出文件
        """
        if result is None:
            print("未检测到交通灯")
            return

        vis_frame = self.source_frame.copy()
        class_index = self.classes.index(result.label)
        color = tuple(int(c) for c in self.color_palette[class_index])

        # 绘制检测框与文本
        cv2.rectangle(vis_frame,
                      (result.x, result.y),
                      (result.x + result.w, result.y + result.h),
                      color, 2)
        cv2.putText(vis_frame, f"{result.label} {result.score:.2f}",
                    (result.x, max(0, result.y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===== 自动生成输出路径 =====
        # 基于脚本所在目录创建 output 文件夹
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 输出文件名与输入图一致（带 "_detected" 后缀）
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        save_path = os.path.join(output_dir, f"{name}_detected{ext}")
        save_path = os.path.normpath(save_path)  # 系统兼容

        # ===== 保存 =====
        success = cv2.imwrite(save_path, vis_frame)
        if success:
            print(f"[INFO] 可视化结果已保存到: {os.path.abspath(save_path)}")
        else:
            print(f"[ERROR] 保存失败，请检查路径权限: {save_path}")


class MultipleDetection:
    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float, class_names: List[str]):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = class_names

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.onnx_model, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_width, self.input_height = input_shape[2], input_shape[3]

    def letterbox(self, img: np.ndarray, new_shape: tuple = (512, 512), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        ratio = r
        pad = (top, left)
        return img, pad, ratio

    def preprocess(self, frame: np.ndarray):
        self.img_height, self.img_width = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, pad, ratio = self.letterbox(img_rgb, (self.input_width, self.input_height))
        self.pad = pad
        self.ratio = ratio
        image_data = np.array(img_lb) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def detect_all(self, image_path: str) -> List[DetectionResult]:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        self.source_frame = frame.copy()    # 保存原始图，后面可视化用
        img_data = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess_all(outputs)

    def postprocess_all(self, output) -> List[DetectionResult]:
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        raw_results: List[DetectionResult] = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0:4]
                # 去除 padding 并映射回原图
                x -= self.pad[1]
                y -= self.pad[0]
                left = int((x - w / 2) / self.ratio)
                top = int((y - h / 2) / self.ratio)
                width = int(w / self.ratio)
                height = int(h / self.ratio)

                raw_results.append(DetectionResult(
                    x=left, y=top, w=width, h=height,
                    label=self.classes[class_id], score=max_score
                ))

        # 执行 NMS 去重
        return self.nms(raw_results, self.iou_thres)

    def nms(self, detections: List[DetectionResult], iou_thres: float) -> List[DetectionResult]:
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self.iou(best, d) < iou_thres]
        return keep

    def iou(self, box1: DetectionResult, box2: DetectionResult) -> float:
        xi1 = max(box1.x, box2.x)
        yi1 = max(box1.y, box2.y)
        xi2 = min(box1.x + box1.w, box2.x + box2.w)
        yi2 = min(box1.y + box1.h, box2.y + box2.h)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = box1.w * box1.h
        box2_area = box2.w * box2.h
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def visualize_on_source(self, results: List[DetectionResult], image_path: str):
        """
        在原图上绘制所有检测结果并保存至 ./output 文件夹（自动适配 Windows/Linux）
        :param results: 检测结果列表
        :param image_path: 输入图片路径（用于自动命名输出文件）
        """
        if not results:
            print("[INFO] 未检测到任何目标")
            return

        vis_frame = self.source_frame.copy()

        # 给不同目标类别随机配置颜色（如未提前定义 color_palette）
        if not hasattr(self, "color_palette"):
            import numpy as np
            self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for res in results:
            class_index = self.classes.index(res.label)
            color = tuple(int(c) for c in self.color_palette[class_index])
            # 绘制矩形框
            cv2.rectangle(vis_frame,
                          (res.x, res.y),
                          (res.x + res.w, res.y + res.h),
                          color, 2)
            # 绘制标签
            cv2.putText(vis_frame,
                        f"{res.label} {res.score:.2f}",
                        (res.x, max(0, res.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # ===== 自动生成输出路径 =====
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 生成以输入图像名为基础的输出文件名
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        save_path = os.path.join(output_dir, f"{name}_detected{ext}")
        save_path = os.path.normpath(save_path)  # 统一路径格式

        # ===== 保存 =====
        success = cv2.imwrite(save_path, vis_frame)
        if success:
            print(f"[INFO] 可视化结果已保存到: {os.path.abspath(save_path)}")
        else:
            print(f"[ERROR] 保存失败，请检查路径或写入权限: {save_path}")
