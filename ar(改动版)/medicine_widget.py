"""
药物识别 UI 界面组件 - 自包含版本
不再依赖 widget_main.create_tool_button 和外部图片文件
ELF+2 开发板 (RK3588) 药物识别系统
"""

import cv2
import numpy as np
from typing import Optional, Dict, List
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea


def _make_btn(text, color_bg="#333333", fixed_size=None):
    """创建自包含的按钮（不依赖外部图片）"""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 18px; padding: 12px 24px;
            background-color: {color_bg}; color: white;
            border-radius: 10px; border: none;
        }}
        QPushButton:hover {{
            background-color: #555555;
        }}
    """)
    if fixed_size:
        btn.setFixedSize(*fixed_size)
    return btn


class MedicineInfoCard(QWidget):
    """药品信息卡片（用于列表中的单个药品展示）"""

    def __init__(self, med_info: Dict, parent=None):
        super().__init__(parent)
        self.med_info = med_info
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        # 药品名称 + 分类标签
        header = QHBoxLayout()
        name_label = QLabel(self.med_info.get('name', '未知药品'))
        name_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00FF88;")
        header.addWidget(name_label)

        category = self.med_info.get('category', '')
        if category:
            cat_label = QLabel(f" [{category}]")
            cat_label.setStyleSheet("font-size: 20px; color: #FFAA00;")
            header.addWidget(cat_label)

        header.addStretch()
        layout.addLayout(header)

        # 规格
        spec = self.med_info.get('specification', '')
        if spec:
            spec_label = QLabel(f"规格: {spec}")
            spec_label.setStyleSheet("font-size: 16px; color: #CCCCCC;")
            layout.addWidget(spec_label)

        # 成分
        ingredients = self.med_info.get('ingredients', [])
        if ingredients:
            ing_text = "成分: " + ", ".join(
                [f"{i.get('ingredient_name', '')}({i.get('quantity', '')})"
                 if i.get('quantity') else i.get('ingredient_name', '')
                 for i in ingredients]
            )
            ing_label = QLabel(ing_text)
            ing_label.setStyleSheet("font-size: 16px; color: #CCCCCC;")
            ing_label.setWordWrap(True)
            layout.addWidget(ing_label)

        # 别名
        aliases = self.med_info.get('aliases', [])
        if aliases:
            alias_label = QLabel(f"别名: {'、'.join(aliases)}")
            alias_label.setStyleSheet("font-size: 16px; color: #999999;")
            layout.addWidget(alias_label)

        # 分隔线
        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #333333;")
        layout.addWidget(line)

        # 适应症
        indications = self.med_info.get('indications', '')
        if indications:
            ind_label = QLabel(f"适应症: {indications}")
            ind_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
            ind_label.setWordWrap(True)
            layout.addWidget(ind_label)

        # 用法用量
        usage = self.med_info.get('usage_dosage', '')
        if usage:
            usage_label = QLabel(f"用法用量: {usage}")
            usage_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
            usage_label.setWordWrap(True)
            layout.addWidget(usage_label)

        # 不良反应
        side = self.med_info.get('side_effects', '')
        if side:
            side_label = QLabel(f"不良反应: {side}")
            side_label.setStyleSheet("font-size: 16px; color: #FF8888;")
            side_label.setWordWrap(True)
            layout.addWidget(side_label)

        # 禁忌
        contra = self.med_info.get('contraindications', '')
        if contra:
            contra_label = QLabel(f"禁忌: {contra}")
            contra_label.setStyleSheet("font-size: 16px; color: #FF6666;")
            contra_label.setWordWrap(True)
            layout.addWidget(contra_label)

        # 注意事项
        caution = self.med_info.get('precautions', '')
        if caution:
            caution_label = QLabel(f"注意事项: {caution}")
            caution_label.setStyleSheet("font-size: 16px; color: #FFAA66;")
            caution_label.setWordWrap(True)
            layout.addWidget(caution_label)

        # 相互作用
        interaction = self.med_info.get('drug_interactions', '')
        if interaction:
            inter_label = QLabel(f"药物相互作用: {interaction}")
            inter_label.setStyleSheet("font-size: 16px; color: #FFAA66;")
            inter_label.setWordWrap(True)
            layout.addWidget(inter_label)

        # 储存条件
        storage = self.med_info.get('storage_condition', '')
        if storage:
            storage_label = QLabel(f"储存: {storage}")
            storage_label.setStyleSheet("font-size: 14px; color: #999999;")
            layout.addWidget(storage_label)

        # 生产厂家
        mfr = self.med_info.get('manufacturer', '')
        if mfr:
            mfr_label = QLabel(f"生产: {mfr}")
            mfr_label.setStyleSheet("font-size: 14px; color: #999999;")
            layout.addWidget(mfr_label)

        # 批准文号
        appr = self.med_info.get('approval_number', '')
        if appr:
            appr_label = QLabel(f"批准文号: {appr}")
            appr_label.setStyleSheet("font-size: 14px; color: #888888;")
            layout.addWidget(appr_label)

    def update_info(self, med_info: Dict):
        """更新显示信息"""
        self.med_info = med_info
        while self.layout().count():
            item = self.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.init_ui()


class MedicineWidget(QWidget):
    """
    药物识别主界面
    完全自包含，不依赖外部图片文件
    """
    medicine_need_signal = pyqtSignal(bool)

    def __init__(self, parent=None, ppocr_instance=None):
        super().__init__(parent)
        self.current_best: Optional[Dict] = None
        self.current_results: List[Dict] = []
        self.camera_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.is_recognizing = False
        self.is_search_mode = False

        self.ppocr = ppocr_instance

        # 性能优化：跳帧
        self._frame_counter = 0
        self._preview_interval = 5
        self._active = False

        self.init_ui()

    def set_ppocr(self, ppocr_instance):
        self.ppocr = ppocr_instance

    def set_active(self, active: bool):
        self._active = active
        self._frame_counter = 0

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(20)

        # ===== 左侧：摄像头画面 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #00FF88;
                border-radius: 10px;
            }
        """)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(True)
        left_layout.addWidget(self.image_label)

        self.status_label = QLabel("将药品包装对准摄像头")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; color: #00FF88; padding: 10px;")
        left_layout.addWidget(self.status_label)

        main_layout.addWidget(left_widget, stretch=1)

        # ===== 右侧：信息面板 =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(10)

        title = QLabel("药品识别结果")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF88;")
        right_layout.addWidget(title)

        # 搜索框
        search_layout = QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("手动输入药品名称搜索...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                font-size: 18px; padding: 10px; border-radius: 8px;
                background-color: #222222; color: white; border: 1px solid #555;
            }
        """)
        self.search_input.returnPressed.connect(self.on_manual_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = _make_btn("搜索", "#0066CC")
        self.search_btn.clicked.connect(self.on_manual_search)
        search_layout.addWidget(self.search_btn)
        right_layout.addLayout(search_layout)

        # 滚动信息区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; }
            QScrollBar:vertical { width: 8px; background: #222; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; }
        """)

        self.info_container = QWidget()
        self.info_layout = QVBoxLayout(self.info_container)
        self.info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_layout.setSpacing(0)

        self.placeholder = QLabel("等待识别...\n\n将药品包装对准摄像头\n或在上方搜索框输入药名")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("font-size: 18px; color: #666666; padding: 40px;")
        self.info_layout.addWidget(self.placeholder)
        self.info_layout.addStretch()

        scroll.setWidget(self.info_container)
        right_layout.addWidget(scroll, stretch=1)

        # 底部按钮行
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.recognize_btn = _make_btn("  拍照识别  ", "#00AA55", (200, 60))
        self.recognize_btn.clicked.connect(self.on_recognize_click)

        self.retake_btn = _make_btn("  重新识别  ", "#CC8800", (200, 60))
        self.retake_btn.clicked.connect(self.reset_view)
        self.retake_btn.setVisible(False)

        self.back_btn = _make_btn("  返回  ", "#555555", (160, 60))

        button_layout.addWidget(self.recognize_btn)
        button_layout.addWidget(self.retake_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.back_btn)

        right_layout.addLayout(button_layout)
        main_layout.addWidget(right_widget, stretch=2)
        self.setStyleSheet("background-color: #111111;")

    def update_camera_frame(self, frame):
        if frame is None:
            return
        self.camera_image = frame.copy()
        if not self._active or self.is_recognizing:
            return
        self._frame_counter += 1
        if self._frame_counter % self._preview_interval != 0:
            return
        try:
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(cv2.flip(small, 1), cv2.COLOR_BGR2RGB)
            self._display_image(rgb)
        except Exception as e:
            print(f"[药物识别] 显示帧失败: {e}")

    def _display_image(self, rgb_img):
        try:
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            self._keep_ref = rgb_img
            q_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"[药物识别] 图像显示错误: {e}")

    def on_recognize_click(self):
        if self.camera_image is None:
            self.status_label.setText("请先等待摄像头画面加载")
            self.status_label.setStyleSheet("font-size: 18px; color: #FF6666; padding: 10px;")
            return
        self.is_recognizing = True
        self.status_label.setText("正在识别...")
        self.status_label.setStyleSheet("font-size: 18px; color: #FFAA00; padding: 10px;")
        best, all_results, annotated = self._run_recognition(self.camera_image)
        self.show_recognition_result(best, all_results, annotated)

    def _run_recognition(self, image):
        """
        识别流程：
        1. PP-OCR 读取药盒文字
        2. 直接在本地数据库搜索（全字段模糊匹配）
        3. 多策略兜底：去后缀、截取前N字、2字分词
        4. 在图上标注OCR识别的文字
        """
        best_match = None
        all_results = []
        annotated = image.copy()

        # Step 1: PP-OCR 识别
        ocr_raw_texts = []
        if self.ppocr is not None:
            try:
                # PP-OCR 需要 RGB 输入，但 camera_image 是 BGR
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, rec_results = self.ppocr.infer(rgb_img)
                if rec_results:
                    for t in rec_results:
                        if isinstance(t, (list, tuple)):
                            txt = t[0] if len(t) > 0 else ""
                            if isinstance(txt, (list, tuple)):
                                txt = txt[0] if len(txt) > 0 else ""
                            ocr_raw_texts.append(str(txt).strip())
                print(f"[药物识别] OCR识别到 {len(ocr_raw_texts)} 段文字: {ocr_raw_texts}")

                # 画框
                if boxes is not None:
                    for box in boxes:
                        try:
                            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                        except Exception:
                            pass
                # 在图上显示OCR文字
                for i, txt in enumerate(ocr_raw_texts[:5]):
                    cv2.putText(annotated, txt, (20, 80 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception as e:
                print(f"[药物识别] OCR推理失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[药物识别] PP-OCR 未加载")

        # Step 2: 直接搜索数据库（不做文本过滤，全部试一遍）
        from medicine_db import get_db
        db = get_db()
        seen_ids = set()

        # 策略1: 直接搜每个OCR文本
        for raw_text in ocr_raw_texts:
            if len(raw_text) < 2:
                continue
            results = db.search_by_name(raw_text)
            for r in results:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    all_results.append(r)
                    if best_match is None:
                        best_match = r
            if best_match:
                break

        # 策略2: 去掉药物后缀再搜（如"胶囊"→去掉）
        if not all_results:
            from medicine_recognizer import DRUG_SUFFIXES
            for raw_text in ocr_raw_texts:
                for suffix in DRUG_SUFFIXES:
                    if raw_text.endswith(suffix) and len(raw_text) > len(suffix):
                        core = raw_text[:-len(suffix)].strip()
                        if len(core) >= 2:
                            results = db.search_by_name(core)
                            for r in results:
                                if r['id'] not in seen_ids:
                                    seen_ids.add(r['id'])
                                    all_results.append(r)
                                    if best_match is None:
                                        best_match = r
                if best_match:
                    break

        # 策略3: 逐渐缩短文本（防止OCR多识别了字符）
        if not all_results:
            for raw_text in ocr_raw_texts:
                for end in range(len(raw_text), 1, -1):
                    sub = raw_text[:end]
                    results = db.search_by_name(sub)
                    for r in results:
                        if r['id'] not in seen_ids:
                            seen_ids.add(r['id'])
                            all_results.append(r)
                            if best_match is None:
                                best_match = r
                    if best_match:
                        break
                if best_match:
                    break

        # 策略4: 每2字滑动切分（应对OCR完全认错一部分字）
        if not all_results:
            for raw_text in ocr_raw_texts:
                for i in range(len(raw_text) - 1):
                    chunk = raw_text[i:i+2]
                    results = db.search_by_name(chunk)
                    for r in results:
                        if r['id'] not in seen_ids:
                            seen_ids.add(r['id'])
                            all_results.append(r)
                            if best_match is None:
                                best_match = r
                if best_match:
                    break

        print(f"[药物识别] 识别结果: {len(all_results)}种, 最佳={best_match['name'] if best_match else '无'}")

        # 在图上标识别结果
        if best_match:
            txt = f"识别: {best_match['name']} ({best_match.get('category', '')})"
            cv2.putText(annotated, txt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "未识别到药品", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 保存图片供调试
        try:
            import os as _os
            _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "medicine_data"), exist_ok=True)
            cv2.imwrite(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "medicine_data", "capture_debug.jpg"), image)
        except Exception:
            pass

        return best_match, all_results, annotated

    def show_recognition_result(self, best, results, annotated):
        self.current_best = best
        self.current_results = results
        self.annotated_image = annotated
        self._clear_info()

        if best is None and not results:
            self.status_label.setText("未识别到药品，可手动搜索")
            self.status_label.setStyleSheet("font-size: 18px; color: #FF6666; padding: 10px;")
            no = QLabel("未识别到药品信息\n\n请确保药品包装清晰可见\n尝试在上方搜索框输入药名")
            no.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no.setStyleSheet("font-size: 18px; color: #FF6666; padding: 40px;")
            self.info_layout.addWidget(no)
            self.info_layout.addStretch()
        else:
            self.status_label.setText(f"识别到 {len(results)} 种药品")
            self.status_label.setStyleSheet("font-size: 18px; color: #00FF88; padding: 10px;")
            for med in results:
                self.info_layout.addWidget(MedicineInfoCard(med))
            self.info_layout.addStretch()

        if annotated is not None:
            try:
                rgb = cv2.cvtColor(cv2.flip(annotated, 1), cv2.COLOR_BGR2RGB)
                self._display_image(rgb)
            except Exception:
                pass
        self.retake_btn.setVisible(True)

    def on_manual_search(self):
        keyword = self.search_input.text().strip()
        if not keyword:
            return
        self.status_label.setText(f"搜索: {keyword}")
        self.status_label.setStyleSheet("font-size: 18px; color: #FFAA00; padding: 10px;")
        self._clear_info()

        from medicine_db import get_db
        db = get_db()
        results = db.search_by_name(keyword)

        if results:
            self.current_results = results
            self.current_best = results[0]
            self.status_label.setText(f"找到 {len(results)} 种药品")
            self.status_label.setStyleSheet("font-size: 18px; color: #00FF88; padding: 10px;")
            for med in results:
                self.info_layout.addWidget(MedicineInfoCard(med))
        else:
            self.current_results = []
            self.current_best = None
            self.status_label.setText("未找到该药品")
            self.status_label.setStyleSheet("font-size: 18px; color: #FF6666; padding: 10px;")
            no = QLabel(f"未找到「{keyword}」的相关信息\n请检查药品名称是否正确")
            no.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no.setStyleSheet("font-size: 18px; color: #FF6666; padding: 40px;")
            self.info_layout.addWidget(no)
        self.info_layout.addStretch()

    def reset_view(self):
        self.is_recognizing = False
        self.current_best = None
        self.current_results = []
        self.annotated_image = None
        self.retake_btn.setVisible(False)
        self.status_label.setText("将药品包装对准摄像头")
        self.status_label.setStyleSheet("font-size: 18px; color: #00FF88; padding: 10px;")
        self._clear_info()
        p = QLabel("等待识别...\n\n将药品包装对准摄像头\n或在上方搜索框输入药名")
        p.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p.setStyleSheet("font-size: 18px; color: #666666; padding: 40px;")
        self.info_layout.addWidget(p)
        self.info_layout.addStretch()

    def _clear_info(self):
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
