import json
import os
from paddleocr import PaddleOCR


# 不许用v5,v5没有车牌数据集
ocr = PaddleOCR(
    text_detection_model_dir="inferences/PP-OCRv5_mobile_det_infer",
    text_recognition_model_name="PP-OCRv4_mobile_rec",
    text_det_limit_side_len=2,
    text_det_box_thresh=0.7,
    text_rec_score_thresh=0.7,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) # 更换 PP-OCRv4_mobile 模型
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv4_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False) # 更换 PP-OCRv4_mobile 模型
result = ocr.predict("./images/license/slot1.png")
# result = ocr.predict("./images/license/license.png")
for res in result:
    res.print()
    res.save_to_img("selfclass/output")
    res.save_to_json("selfclass/output")

# 生成的 OCR JSON 文件路径
# json_path = os.path.join("selfclass/output", "license_res.json")
json_path = os.path.join("selfclass/output", "slot1_res.json")

# 读取 JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

rec_texts = data.get("rec_texts", [])
rec_scores = data.get("rec_scores", [])

plate_number = None
plate_score = None

# 找出第一个有效字符串
for text, score in zip(rec_texts, rec_scores):
    if text.strip():  # 非空
        plate_number = text.strip()
        plate_score = score
        break

# 输出结果
if plate_number:
    print(f"识别到车牌号: {plate_number}, 置信度: {plate_score:.3f}")
else:
    print("未识别到车牌号")
