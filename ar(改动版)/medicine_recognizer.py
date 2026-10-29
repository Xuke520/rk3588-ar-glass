"""
药物识别核心模块
使用 PP-OCR 读取药品包装上的文字 → 模糊匹配数据库 → 返回药品信息
ELF+2 开发板 (RK3588) 药物识别系统
"""

import re
from typing import List, Dict, Optional, Tuple

# OpenCV/Numpy 可选导入（文本提取依赖）
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
from medicine_db import get_db


# ===== 药品名称关键词库（用于从OCR结果中提取药名）=====
# 常见药品后缀/特征词
DRUG_SUFFIXES = [
    "胶囊", "片", "软胶囊", "口服液", "颗粒", "滴丸", "注射剂",
    "注射液", "片剂", "冲剂", "糖浆", "合剂", "散剂", "丸",
    "膏", "贴", "喷雾", "气雾剂", "滴眼液", "滴眼剂",
    "软膏", "乳膏", "凝胶", "栓剂", "泡腾片", "缓释片",
    "控释片", "肠溶片", "分散片", "咀嚼片", "含片", "漱口水"
]

# 需过滤的常见非药名词汇
STOP_WORDS = [
    "国药准字", "批准文号", "生产日期", "有效期", "生产批号",
    "产品批号", "贮藏", "储存", "密封", "阴凉", "遮光",
    "用法用量", "不良反应", "禁忌", "注意事项", "适应症",
    "规格", "每板", "每盒", "包装", "企业", "地址",
    "说明书", "请仔细阅读", "OTC", "RX", "处方药", "非处方药",
    "otc", "www", "com", "有限公司", "药业", "制药",
    "电话", "传真", "邮编", "网址", "服务热线", "免费电话",
    "一日", "一次", "口服", "外用", "mg", "g", "ml",
    "片*", "粒*", "支*", "瓶*", "盒*",
    # 常见干扰词
    "副作用", "药物相互作用", "儿童", "成人", "老年",
    "孕妇", "哺乳期", "肝", "肾", "过敏", "禁用", "慎用",
    "乙醇", "辅料", "成分", "性状"
]


def extract_medicine_names(texts: List[str]) -> List[str]:
    """
    从OCR识别的文本列表中提取可能的药品名称
    返回按置信度排序的候选药名列表
    """
    candidates = []

    for text in texts:
        text = text.strip()

        # 跳过空文本和过短文本
        if len(text) < 2:
            continue

        # 跳过停止词
        if any(stop in text for stop in STOP_WORDS):
            continue

        # 跳过纯数字/字母
        if re.match(r'^[\d\.]+$', text):
            continue

        # 检查是否包含药品后缀（高置信度候选）
        has_suffix = any(text.endswith(suffix) for suffix in DRUG_SUFFIXES)

        # 检查是否是常见药名字典里的词
        if has_suffix:
            candidates.append((text, 1.0))
        else:
            candidates.append((text, 0.5))

    # 按置信度排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates]


def search_medicine(texts: List[str]) -> Tuple[Optional[Dict], List[Dict]]:
    """
    对OCR结果进行药品搜索
    返回: (最佳匹配, 所有匹配列表)
    """
    db = get_db()
    candidates = extract_medicine_names(texts)

    all_results = []
    best_match = None
    seen_ids = set()

    for name in candidates:
        results = db.search_by_name(name)
        for r in results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                all_results.append(r)
                if best_match is None:
                    # 如果OCR文本直接命中或包含药物后缀，作为最佳匹配
                    if r['name'] == name or any(name.endswith(s) for s in DRUG_SUFFIXES):
                        best_match = r

    # 如果没找到精确匹配，取第一个结果
    if best_match is None and all_results:
        best_match = all_results[0]

    return best_match, all_results


class MedicineRecognizer:
    """
    药物识别器
    整合PP-OCR + 数据库搜索 + 图像预处理
    """

    def __init__(self, ppocr_instance=None):
        """
        ppocr_instance: PPOCRRKNN 实例（如果为None，使用模拟模式）
        """
        self.ppocr = ppocr_instance
        self.db = get_db()
        if not _HAS_CV2:
            print("[药物识别] 警告: OpenCV 未安装，图像预处理和标注功能不可用")

    def preprocess_image(self, image) -> any:
        """图像预处理：提升OCR识别率"""
        if not _HAS_CV2:
            return image

        img = image.copy()

        # 1. 转为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. 自适应直方图均衡化（增强对比度）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. 去噪
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # 4. 转回BGR（PP-OCR需要3通道）
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return result

    def recognize(self, image) -> Tuple[Optional[Dict], List[Dict], any]:
        """
        识别药品包装图像
        参数:
            image: BGR图像
        返回:
            (最佳匹配药品信息, 所有匹配列表, 标注后的图像)
        """
        if image is None:
            return None, [], image

        # 预处理
        processed = self.preprocess_image(image)

        # OCR识别
        boxes, texts = self._run_ocr(processed)

        # 搜索结果
        ocr_texts = [t[0][0] if isinstance(t, (list, tuple)) else str(t) for t in (texts or [])]

        best_match, all_results = search_medicine(ocr_texts)

        # 在图像上标注结果
        annotated = self._draw_results(image, boxes, texts, best_match)

        return best_match, all_results, annotated

    def _run_ocr(self, image) -> Tuple[List, List]:
        """运行OCR识别"""
        if self.ppocr is not None and _HAS_CV2:
            try:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, rec_results = self.ppocr.infer(rgb_img)
                return boxes or [], rec_results or []
            except Exception as e:
                print(f"[药物识别] OCR推理失败: {e}")
                return [], []
        else:
            # 模拟模式：无PP-OCR时的备选
            print("[药物识别] 警告: PP-OCR未加载，使用模拟识别模式")
            return [], []

    def _draw_results(self, image, boxes: List,
                      texts: List, best_match: Optional[Dict]) -> any:
        """在图像上标注识别结果"""
        if not _HAS_CV2:
            return image
        img = image.copy()

        # 绘制OCR文本框
        if boxes is not None:
            for box in boxes:
                try:
                    box_pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [box_pts], True, (0, 255, 0), 2)
                except Exception:
                    pass

        # 在顶部显示识别结果
        if best_match:
            text = f"识别: {best_match['name']} ({best_match['category']})"
            # 半透明背景
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

    def recognize_from_camera_frame(self, frame):
        """从摄像头帧直接识别"""
        return self.recognize(frame)
