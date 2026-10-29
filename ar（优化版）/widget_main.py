import os
import sys
import cv2
import numpy as np
from PyQt6 import QtCore,QtGui,QtWidgets
from PyQt6.QtWidgets import QApplication,QWidget,QGridLayout,QPushButton,QHBoxLayout,QVBoxLayout,QLabel,QMessageBox,QFileDialog,QLineEdit
from PyQt6.QtCore import Qt,QTimer,pyqtSignal,QThread,QDir
from PyQt6.QtGui import QPainter,QPen,QColor,QPixmap
from hand_gesture import hand_connections
import time
from neural_net.rknn.ppocr.ppocr import PPOCRRKNN
from neural_net.rknn.yolov5.yolov5 import Yolov5RKNN

from PIL import Image, ImageDraw, ImageFont


class MainContentWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        #主垂直布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 50, 50, 50)  # 保留边距
        
        # 添加顶部弹簧（实现垂直居中）
        main_layout.addStretch(1)
        
        # 上部按钮行（3个按钮）
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 50)
        top_layout.setSpacing(100)
        
        # 下部按钮行（2个按钮）
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 50, 0, 0)
        bottom_layout.setSpacing(100)
        
        # 按钮配置
        button_configs = [
            {"row": "top", "id": "text_reg", "text": "文字识别", "icon": "text.png"},
            {"row": "top", "id": "speech_reg", "text": "语音识别", "icon": "speech.png"},
            {"row": "top", "id": "remote_ctrl", "text": "远程控制", "icon": "remote.png"},
            {"row": "bottom", "id": "remote_camera", "text": "远程串流", "icon": "camera.png"},
            {"row": "bottom", "id": "object_detection", "text": "目标检测", "icon": "detect.png"}
        ]

        self.buttons = {}
        
        # 创建按钮
        for config in button_configs:
            btn = create_tool_button(config["text"], config["icon"], 200)
            btn.setFixedSize(280, 280)
            btn.setObjectName(config["id"])
            self.buttons[config["id"]] = btn
            
            if config["row"] == "top":
                top_layout.addWidget(btn)
            else:
                bottom_layout.addWidget(btn)
        
        # 添加上下行的居中弹簧
        top_layout.insertStretch(0, 1)  # 左侧弹簧
        top_layout.addStretch(1)       # 右侧弹簧
        bottom_layout.insertStretch(0, 1)
        bottom_layout.addStretch(1)
        
        # 将上下部分添加到主布局（居中）
        main_layout.addWidget(top_widget, 0, Qt.AlignmentFlag.AlignCenter)
        main_layout.addSpacing(30)  # 上下行间距
        main_layout.addWidget(bottom_widget, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 添加底部弹簧（实现垂直居中）
        main_layout.addStretch(1)

class ModelWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_file_path = None
        self.init_ui()

    def init_ui(self):
        # 主水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(20)

        button_layout=QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.switch_btn=create_tool_button("选择",'switch',150)
        self.smooth_btn=create_tool_button("平滑",'smooth',150)
        self.grid_btn=create_tool_button("网格",'grid',150)
        self.back_btn=create_tool_button("返回",'back.png',150)
        self.switch_btn.setFixedSize(250, 200)
        self.smooth_btn.setFixedSize(250, 200)
        self.grid_btn.setFixedSize(250, 200)
        self.back_btn.setFixedSize(250, 200)

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch()
        button_layout.addWidget(self.switch_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.smooth_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.grid_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addStretch()

        # 将按钮列添加到主布局
        main_layout.addLayout(button_layout, stretch=0)

        self.switch_btn.clicked.connect(self.switch_file)

    def switch_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("选择文件")
        file_dialog.setDirectory(QDir.homePath())  # 默认从用户主目录开始
        file_dialog.setNameFilter("All Files (*);;Text Files (*.txt);;Images (*.png *.jpg)")
        
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.selected_file_path = selected_files[0]
                print(f"已选择文件: {self.selected_file_path}")
   
class TextRegWidget(QWidget):
    text_need_signal = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label=None
        self.ppocr_running=False
        self.ppocr=PPOCRRKNN()
        self.init_ui()
    
    def init_ui(self):
        # 主水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(200, 10, 20, 10)
        main_layout.setSpacing(20)

        # 左侧图片标签
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
        """)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(True)
        main_layout.addWidget(self.image_label, stretch=1)

        # 右侧按钮列 (垂直布局)
        button_layout=QVBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 返回按钮
        self.back_btn=create_tool_button("返回",'back.png',150)
        self.back_btn.setFixedSize(250, 200)

        # 翻译按钮
        self.translate_btn=create_tool_button("翻译",'translate.png',150)
        self.translate_btn.setFixedSize(250, 200)

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch()
        button_layout.addWidget(self.translate_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addSpacing(50)
        button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addStretch()

        # 将按钮列添加到主布局
        main_layout.addLayout(button_layout, stretch=0)

    def draw_ocr_results(self,image, boxes, texts):
        """将识别文字绘制在文本框内部的解决方案"""
        # 转换为PIL格式
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # 字体加载（保持原有逻辑）
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            os.path.join(os.path.dirname(__file__), "fonts/simhei.ttf")
        ]
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 20)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # 统一使用PIL绘制
        for box, text in zip(boxes, texts):
            box = [tuple(p.astype(int)) for p in box]
            
            # 绘制边界框（绿色）
            draw.polygon(box, outline=(0, 255, 0), width=2)
            
            # 处理文本内容
            text_content = str(text[0][0]).encode('utf-8').decode('utf-8', 'ignore')
            
            # 计算文本框的边界
            min_x = min(p[0] for p in box)
            max_x = max(p[0] for p in box)
            min_y = min(p[1] for p in box)
            max_y = max(p[1] for p in box)
            
            # 计算文本位置（居中显示）
            left, top, right, bottom = font.getbbox(text_content)
            text_width = right - left
            text_height = bottom - top
            text_x = min_x + (max_x - min_x - text_width) // 2
            text_y = min_y + (max_y - min_y - text_height) // 2
            
            # 确保文本不会超出框边界
            text_x = max(min_x, min(text_x, max_x - text_width))
            text_y = max(min_y, min(text_y, max_y - text_height))
            
            # 绘制半透明背景（增强可读性）
            bg_box = (
                text_x - 2, text_y - 2,
                text_x + text_width + 2,
                text_y + text_height + 2
            )
            bg_layer = Image.new('RGBA', image_pil.size, (0,0,0,0))
            bg_draw = ImageDraw.Draw(bg_layer)
            bg_draw.rectangle(bg_box, fill=(0, 0, 255, 128))  # 半透明蓝色背景
            image_pil = Image.alpha_composite(image_pil.convert('RGBA'), bg_layer)
            draw = ImageDraw.Draw(image_pil)
            
            # 绘制文本（白色）
            draw.text((text_x, text_y), text_content, font=font, fill=(255, 255, 255))
        
        return cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

    def update_text(self,img):
        if img is None:
            return 
        rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w, ch=rgb_img.shape
        
        
        if self.ppocr_running:
            print("OCR processing...")  
            filter_boxes,filter_rec_res=self.ppocr.infer(rgb_img)
            if filter_boxes is not None and filter_rec_res is not None:
                result_frame=self.draw_ocr_results(rgb_img.copy(),filter_boxes,filter_rec_res)
        else:
            result_frame=rgb_img.copy()

            # for i, box in enumerate(filter_boxes):
            #     # 绘制检测框
            #     box = box.astype(int)
            #     cv2.polylines(rgb_img, [box], True, (0, 255, 0), 2)
                
            #     # 获取识别结果
            #     if i < len(filter_rec_res):
            #         rec_result = filter_rec_res[i]
                    
            #         # 处理不同的返回格式
            #         if isinstance(rec_result, (tuple, list)) and len(rec_result) >= 1:
            #             text = str(rec_result[0])  # 确保转为字符串
            #             score = rec_result[1] if len(rec_result) > 1 else 0.0
            #         else:
            #             text = str(rec_result)
            #             score = 0.0
                    
            #         # 清理文本（移除可能的分隔符）
            #         text = text.strip()
            #         if text and text != '':
            #             # 确定文本位置（使用检测框的左上角）
            #             text_position = (int(box[0][0]), int(box[0][1]) - 5)
                        
            #             # 如果位置超出图像边界，调整到框内
            #             if text_position[1] < 0:
            #                 text_position = (int(box[0][0]), int(box[0][1]) + 20)
                        
            #             # 绘制文本
            #             try:
            #                 cv2.putText(rgb_img, text, text_position,
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #             except Exception as e:
            #                 print(f"绘制文本失败: {e}, text={text}")
                


        bytes_per_line =ch * w
        q_image = QtGui.QImage(
            result_frame.tobytes(), 
            w, h, 
            bytes_per_line, 
            QtGui.QImage.Format.Format_RGB888
        )
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_image))


class SpeechRegWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.speech_thread = SpeechRegThread()
        # self.speech_thread.speech_signal.connect(self.update_speech)
        # self.speech_thread.start()
        self.init_ui()

    def init_ui(self):
        # 创建主布局（保持与原来相同的布局结构）
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建返回按钮（保持原有样式）
        self.back_btn = create_tool_button("返回",'back.png',150)
        self.back_btn.setFixedSize(250, 200)

        # 创建语音识别结果显示容器（保持原有样式和属性）
        self.speech_container = QtWidgets.QTextEdit()
        self.speech_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speech_container.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: white;
                border: 2px solid gray;
                border-radius: 10px;
                font-size: 50px;
                padding: 30px;
            }
        """)
        self.speech_container.setReadOnly(True)
        self.speech_container.setFixedHeight(200)
        self.speech_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # 添加控件到布局（保持原有比例）
        main_layout.addWidget(self.back_btn, stretch=0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.speech_container, stretch=5)
    
    # def showEvent(self,event):
    #     #当widget显示时激活识别
    #     self.speech_thread.set_running(True)
    #     super().showEvent(event)
    
    # def hideEvent(self,event):
    #     #当widget隐藏时暂停识别
    #     self.speech_thread.set_running(False)
    #     super().hideEvent(event)

    # def update_speech(self,speech):
    #     if speech:        
    #         self.speech_container.setText(speech)
    #         self.speech_container.verticalScrollBar().setValue(
    #             self.speech_container.verticalScrollBar().maximum()
    #         )

class RemoteCtrlWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # 主垂直布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(40)

        # 主控制区域（水平布局）
        control_area = QWidget()
        control_layout = QHBoxLayout(control_area)
        control_layout.setContentsMargins(0, 50, 0, 0)
        control_layout.setSpacing(60)

        # ===== 左侧：灯控制 =====
        lamp_widget = QWidget()
        lamp_layout = QHBoxLayout(lamp_widget)
        lamp_layout.setSpacing(30)

        # 灯状态图标（会随开关变化）
        self.lamp_icon = QLabel()
        self.lamp_icon.setPixmap(QPixmap("/home/elf/work/ar/image/lamp_off.png").scaled(150, 150, 
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation))
        self.lamp_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lamp_icon.setFixedSize(250, 250)

        # 灯控制按钮（垂直排列）
        lamp_btn_layout = QVBoxLayout()
        lamp_btn_layout.setSpacing(20)
        
        self.lamp_on_btn = create_tool_button("开启", "on.png", 150)
        self.lamp_off_btn = create_tool_button("关闭", "off.png", 150)
        for btn in [self.lamp_on_btn, self.lamp_off_btn]:
            btn.setFixedSize(250, 200)
            lamp_btn_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        lamp_btn_layout.addStretch()
        lamp_layout.addWidget(self.lamp_icon)
        lamp_layout.addLayout(lamp_btn_layout)

        # ===== 右侧：风扇控制 =====
        fan_widget = QWidget()
        fan_layout = QHBoxLayout(fan_widget)
        fan_layout.setSpacing(30)

        self.fan_icon = QLabel()
        self.fan_icon.setPixmap(QPixmap("/home/elf/work/ar/image/fan.png").scaled(150, 150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation))
        self.fan_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fan_icon.setFixedSize(250, 250)

        fan_btn_layout = QVBoxLayout()
        fan_btn_layout.setSpacing(15)
        
        speed_btns = [
            ("加速", "jia.png"),
            ("减速", "jian.png"),
            ("关闭", "off.png")
        ]
        
        self.fan_speed_btns = []
        for text, icon in speed_btns:
            btn = create_tool_button(text, icon, 120)
            btn.setFixedSize(250, 200)
            fan_btn_layout.addWidget(btn)
            self.fan_speed_btns.append(btn)
        
        fan_btn_layout.addStretch()
        fan_layout.addWidget(self.fan_icon)
        fan_layout.addLayout(fan_btn_layout)

        # 将左右两部分添加到控制区域
        control_layout.addWidget(lamp_widget, 1)
        control_layout.addWidget(fan_widget, 1)

        # ===== 底部返回按钮 =====
        self.back_btn = create_tool_button("返回", "back.png", 100)
        self.back_btn.setFixedSize(200, 150)

        # ===== 组合所有部分 =====
        main_layout.addWidget(control_area, 1)
        main_layout.addWidget(self.back_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # 设置样式
        self.setStyleSheet("""
            QLabel {
                background: transparent;
            }
            QToolButton {
                font-size: 14px;
                border-radius: 8px;
                padding: 5px;
            }
        """)

        self.lamp_on_btn.clicked.connect(lambda: self.set_lamp_status(True))
        self.lamp_off_btn.clicked.connect(lambda: self.set_lamp_status(False))

    def set_lamp_status(self, on):
        """更新灯状态"""
        self.lamp_on = on
        self.update_lamp_icon()
        
        # 这里可以添加实际控制灯的代码
        # 例如: self.control_lamp(on)
    
    def update_lamp_icon(self):
        """更新灯图标显示"""
        icon_path = "/home/elf/work/ar/image/lamp_on.png" if self.lamp_on else "/home/elf/work/ar/image/lamp_off.png"
        pixmap = QPixmap(icon_path).scaled(
            120, 120,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lamp_icon.setPixmap(pixmap)
             
# class RemoteCameraWidget(QWidget):
#     rcam_need_signal=pyqtSignal(bool)
#     def __init__(self,parent=None):
#         super().__init__(parent)
#         self.init_ui()
#         self.rcam='http://10.30.236.200:8080/video'
#         self.cap=cv2.VideoCapture(self.rcam)

#     def init_ui(self):
#         # 主水平布局
#         main_layout = QHBoxLayout(self)
#         main_layout.setContentsMargins(200, 10, 20, 10)
#         main_layout.setSpacing(20)

#         # 左侧图片标签
#         self.image_label = QtWidgets.QLabel()
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.image_label.setStyleSheet("""
#             QLabel {
#                 background-color: black;
#                 border: 2px solid gray;
#                 border-radius: 10px;
#             }
#         """)
#         self.image_label.setFixedSize(640, 480)
#         self.image_label.setScaledContents(True)
#         main_layout.addWidget(self.image_label, stretch=1)

#         # 右侧按钮列 (垂直布局)
#         button_layout=QVBoxLayout()
#         button_layout.setSpacing(20)
#         button_layout.setContentsMargins(0, 0, 0, 0)

#         # 返回按钮
#         self.back_btn=create_tool_button("返回",'back.png',150)
#         self.back_btn.setFixedSize(250, 200)

#         # 添加弹簧使按钮垂直居中
#         button_layout.addStretch()
#         button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
#         button_layout.addStretch()

#         # 将按钮列添加到主布局
#         main_layout.addLayout(button_layout, stretch=0)

#     def update_frame(self):
#         if self.cap.isOpened():
#             print("错误：无法打开摄像头")
#             return
#         ret,frame=self.cap.read()
#         cv2.imshow('Camera Test', frame)



class RemoteCameraWidget(QWidget):
    rcam_need_signal = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.is_streaming = False
        self.default_url = 'http://192.168.1.8:8080/video'
        self.init_ui()
        self.setup_timer()
    
    def init_ui(self):
        # 主水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(200, 10, 20, 10)
        main_layout.setSpacing(20)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # 左侧图片标签
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
        """)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(True)
        left_layout.addWidget(self.image_label)

        # 流地址输入和连接按钮
        url_layout = QHBoxLayout()
        url_layout.setSpacing(10)
        self.stream_url_input = QLineEdit(self)
        self.stream_url_input.setPlaceholderText('请输入手机视频流地址，例如 http://192.168.1.10:8080/video 或 rtsp://192.168.1.10:8554/live')
        self.stream_url_input.setText(self.default_url)
        self.connect_btn = QPushButton('连接', self)
        self.connect_btn.setFixedHeight(40)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        url_layout.addWidget(self.stream_url_input)
        url_layout.addWidget(self.connect_btn)
        left_layout.addLayout(url_layout)

        main_layout.addLayout(left_layout, stretch=1)

        # 右侧按钮列 (垂直布局)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 返回按钮
        self.back_btn = create_tool_button("返回", 'back.png', 150)
        self.back_btn.setFixedSize(250, 200)
        self.back_btn.clicked.connect(self.stop_streaming)

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch()
        button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addStretch()

        # 将按钮列添加到主布局
        main_layout.addLayout(button_layout, stretch=0)
        
        # 连接信号
        self.rcam_need_signal.connect(self.on_streaming_toggle)
    
    def setup_timer(self):
        """设置定时器用于更新视频帧"""
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(33)  # 约30帧/秒
    
    def on_streaming_toggle(self, is_start):
        """处理远程串流开关信号"""
        if is_start:
            self.start_streaming(self.stream_url_input.text())
        else:
            self.stop_streaming()
    
    def on_connect_clicked(self):
        url = self.stream_url_input.text().strip()
        self.start_streaming(url)

    def start_streaming(self, url: str = None):
        """开始远程串流"""
        if self.is_streaming:
            self.stop_streaming()

        if not url:
            self.show_error_message("请先输入有效的流地址")
            return

        self.stream_url_input.setText(url)
        self.image_label.setText('正在连接...')
        def try_open(u):
            try:
                cap = cv2.VideoCapture(u)
                if cap is not None and cap.isOpened():
                    return cap
            except Exception:
                pass

            # 如果 OpenCV 支持 FFMPEG backend，尝试显式使用它
            if hasattr(cv2, 'CAP_FFMPEG'):
                try:
                    cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
                    if cap is not None and cap.isOpened():
                        return cap
                except Exception:
                    pass
            return None

        try:
            # 1) 先尝试原始 URL
            self.cap = try_open(url)

            # 2) 如果是 https 协议，尝试回退为 http（很多手机摄像头只支持 http）
            if self.cap is None and url.lower().startswith('https://'):
                http_url = 'http://' + url[len('https://'):]
                self.cap = try_open(http_url)
                if self.cap is not None:
                    url = http_url

            if self.cap is None:
                print(f"错误：无法连接到远程摄像头 {url}")
                self.show_error_message(f"无法连接到远程摄像头: {url}\n请确认使用 http 协议并在浏览器/ffplay 中可访问。")
                return

            self.is_streaming = True
            self.timer.start()
            print(f"远程串流已启动: {url}")

        except Exception as e:
            print(f"启动远程串流失败: {e}")
            self.show_error_message(f"启动失败: {str(e)}")
    
    def stop_streaming(self):
        """停止远程串流"""
        if self.timer.isActive():
            self.timer.stop()
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        self.is_streaming = False
        
        # 清空图片显示
        self.image_label.clear()
        self.image_label.setText("串流已停止")
        print("远程串流已停止")
    
    def update_frame(self):
        """更新视频帧显示"""
        if not self.is_streaming or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # 转换颜色空间并显示
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_image = QtGui.QImage(
                    rgb_frame.data, w, h, bytes_per_line,
                    QtGui.QImage.Format.Format_RGB888
                )
                self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_image))
            else:
                # 读取失败时停止串流
                print("视频流中断")
                self.stop_streaming()
                self.show_error_message("视频流已中断")
                
        except Exception as e:
            print(f"更新视频帧失败: {e}")
            self.stop_streaming()
    
    def show_error_message(self, message):
        """显示错误消息"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("错误")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def closeEvent(self, event):
        """确保在关闭窗口时释放资源"""
        self.stop_streaming()
        event.accept()

class ObjectDetectionWidget(QWidget):
    od_need_signal = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = None
        self.yolo_running = False
        self.yolo = Yolov5RKNN()  # 初始化时创建模型实例
        self.init_ui()
    
    def init_ui(self):
        # 主水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(200, 10, 20, 10)
        main_layout.setSpacing(20)

        # 左侧图片标签
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
        """)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(True)
        main_layout.addWidget(self.image_label, stretch=1)

        # 右侧按钮列 (垂直布局)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 返回按钮
        self.back_btn = create_tool_button("返回", "back.png", 150)
        self.back_btn.setFixedSize(250, 200)

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch()
        button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addStretch()

        # 将按钮列添加到主布局
        main_layout.addLayout(button_layout, stretch=0)

    def showEvent(self, event):
        """当widget显示时自动启动检测"""
        self.start_detection()
        super().showEvent(event)

    def hideEvent(self, event):
        """当widget隐藏时自动停止检测"""
        self.stop_detection()
        super().hideEvent(event)

    def start_detection(self):
        """开始目标检测"""
        try:
            self.yolo_running = True
            self.od_need_signal.emit(True)
            print("目标检测已启动")
        except Exception as e:
            print(f"启动目标检测失败: {e}")
            self.show_error_message(f"启动失败: {str(e)}")

    def stop_detection(self):
        """停止目标检测"""
        self.yolo_running = False
        self.od_need_signal.emit(False)
        print("目标检测已停止")

    def update_frame(self, img):
        """更新视频帧并执行检测（与文字识别模块相同的模式）"""
        if img is None:
            return
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape

        if self.yolo_running:
            print("Object detection processing...")
            # 执行目标检测（只识别人）
            result_frame = self.yolo.infer(img.copy())
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        else:
            result_frame = rgb_img.copy()

        # 转换并显示图像
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            result_frame.data, w, h, bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        )
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def show_error_message(self, message):
        """显示错误消息"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("错误")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def closeEvent(self, event):
        """确保在关闭窗口时释放资源"""
        self.stop_detection()
        if self.yolo is not None:
            self.yolo.release()
        event.accept()

class PhotoWidget(QWidget):
    photo_need_signal=pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_editing = False
        self.frozen_image = None
        self.original_style = None
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(200, 10, 20, 10)
        main_layout.setSpacing(20)

        # 左侧图片标签
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
        """)
        self. image_label.setFixedSize(640, 480)
        self. image_label.setScaledContents(True)
        main_layout.addWidget(self. image_label, stretch=1)

        # 右侧按钮列 (垂直布局)
        button_layout=QVBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 编辑按钮
        self.edit_btn = create_tool_button("编辑","edit.png",150)
        self.edit_btn.setFixedSize(250, 200)
        self.edit_btn.clicked.connect(self.toggle_edit_mode)
        
        # 保存按钮
        self.save_btn = create_tool_button("保存","save.png",150)
        self.save_btn.setFixedSize(250, 200)
        self.save_btn.clicked.connect(self.save_image)
        
        # 返回按钮
        self.back_btn = create_tool_button("返回","back.png",150)
        self.back_btn.setFixedSize(250, 200)
        self.back_btn.clicked.connect(self.reset_photo_view)

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch()
        button_layout.addWidget(self.edit_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addSpacing(50)
        button_layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addSpacing(50)
        button_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addStretch()

        # 将按钮列添加到主布局
        main_layout.addLayout(button_layout, stretch=0)
    
    def toggle_edit_mode(self):
        """切换编辑模式"""
        self.is_editing = not self.is_editing
        if self.is_editing:
            if hasattr(self, 'current_image'):
                self.frozen_image = self.current_image.copy()
                self.update_photo(self.frozen_image)
            self.edit_btn.setText("退出编辑")
        else:
            self.frozen_image = None
            self.edit_btn.setText("编辑")
    
    def update_photo(self, img):
        """更新显示的图片"""
        if img is None:
            self.reset_photo_view()
            return
        
        self.current_image = img.copy()
        
        # 使用冻结图像（如果处于编辑模式）
        if self.is_editing and self.frozen_image is not None:
            img = self.frozen_image
        
        # 转换并显示图像
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            rgb_img.data, w, h, bytes_per_line, 
            QtGui.QImage.Format.Format_RGB888
        )
        self. image_label.setPixmap(QtGui.QPixmap.fromImage(q_image))
    
    def save_image(self):
        """保存当前图片"""
        if self.frozen_image is not None:
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            file_path = f"photo_{timestamp}.png"
            cv2.imwrite(file_path, cv2.cvtColor(self.frozen_image, cv2.COLOR_RGB2BGR))
            
            self.msg_label = QtWidgets.QLabel(self)
            self.msg_label.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | 
                                        QtCore.Qt.WindowType.WindowStaysOnTopHint)
            self.msg_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.msg_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(50, 50, 50, 180);
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                    font-size: 24px;
                    min-width: 300px;
                }
            """)
            self.msg_label.setText(f"✓ 图片已保存\n{file_path}")
            self.msg_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  
            # 定位到父窗口中央
            parent_rect = self.geometry()
            msg_width = 350
            msg_height = 100

            # 计算位置（底部居中，距离底部20像素）
            x = parent_rect.x() + (parent_rect.width() - msg_width) // 2
            y = parent_rect.y() + parent_rect.height() - msg_height - 20  # 距离底部20px

            self.msg_label.setGeometry(x, y, msg_width, msg_height)
            self.msg_label.show()
            # 设置淡出动画
            self.animation = QtCore.QPropertyAnimation(self.msg_label, b"windowOpacity")
            self.animation.setDuration(1000)  # 1秒动画
            self.animation.setStartValue(1.0)
            self.animation.setEndValue(0.0)
            self.animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)
            # 淡出
            QtCore.QTimer.singleShot(1500, self.start_fade_out)

    def start_fade_out(self):
        """启动淡出动画"""
        self.animation.start()
        self.animation.finished.connect(self.msg_label.deleteLater)
    
    def reset_photo_view(self):
        """重置图片视图"""
        self.image_label.clear()
        self.image_label.setPixmap(QtGui.QPixmap())
        self.image_label.setStyleSheet(self.original_style)
        self.is_editing = False
        self.frozen_image = None
        if hasattr(self, 'current_image'):
            del self.current_image
        self.photo_need_signal.emit(False)

    def handle_gesture_drawing(self, landmarks):
        """处理手势绘制 - 只有当手势在图片范围内才绘制"""
        if not self.is_editing or self.frozen_image is None:
            return
            
        if len(landmarks) > 8:  # 确保有食指关键点
            # 获取图片控件的位置和大小
            img_rect = self. image_label.geometry()
            img_x, img_y = img_rect.x(), img_rect.y()
            img_width, img_height = img_rect.width(), img_rect.height()
            
            # 获取食指在窗口中的绝对位置
            index_tip = landmarks[8]
            window_x = int(index_tip[0] * self.width())
            window_y = int(index_tip[1] * self.height())
            
            # 转换为相对于图片控件的位置
            rel_x = window_x - img_x
            rel_y = window_y - img_y
            
            # 检查是否在图片范围内
            if 0 <= rel_x < img_width and 0 <= rel_y < img_height:
                # 转换为图片坐标
                img_coord_x = int(rel_x * (self.frozen_image.shape[1] / img_width))
                img_coord_y = int(rel_y * (self.frozen_image.shape[0] / img_height))
                
                # 在图像上绘制
                cv2.circle(self.frozen_image, (img_coord_x, img_coord_y), 10, (0, 255, 0), -1)
                self.update_photo(self.frozen_image)

def create_tool_button(text, icon_path, icon_height, parent=None):
    button = QtWidgets.QPushButton(parent)
    button.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)  # 启用触摸事件
    button.setStyleSheet("""
        QPushButton {
            background-color: rgba(0,0,0,150);
            border-radius: 10px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #555;
        }
    """)
    
    layout = QtWidgets.QVBoxLayout(button)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(5)
    
    icon = QtWidgets.QLabel()
    icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    pixmap = QtGui.QPixmap(f"/home/elf/work/ar/image/{icon_path}")
    icon.setPixmap(pixmap.scaled(icon_height, icon_height, 
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation))
    
    label = QtWidgets.QLabel(text)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
    """)
    
    layout.addWidget(icon)
    layout.addWidget(label)
    
    return button

class GestureLayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |  # 无边框
            Qt.WindowType.WindowStaysOnTopHint   # 始终在最上层
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)  # 关键修复
        self.pen = QPen(QColor(255, 50, 50, 255), 8, Qt.PenStyle.DashLine)
        self.landmarks = None
 
    def update_gesture(self, landmarks):
        self.landmarks = landmarks
        self.update()
 
    def paintEvent(self,event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.landmarks:
            width, height = self.width(), self.height()
            painter.setPen(self.pen)
            
            # 绘制关键点
            for landmark in self.landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                painter.drawEllipse(x, y, 5, 5)
            
            # 绘制连接线
            for connection in hand_connections:
                if (connection[0] < len(self.landmarks) and (connection[1] < len(self.landmarks))):
                    x1 = int(self.landmarks[connection[0]][0] * width)
                    y1 = int(self.landmarks[connection[0]][1] * height)
                    x2 = int(self.landmarks[connection[1]][0] * width)
                    y2 = int(self.landmarks[connection[1]][1] * height)
                    painter.drawLine(x1, y1, x2, y2)
        
        painter.end()

class MainWindow(QWidget):
    func_signal=pyqtSignal(int)
    update_speech_text = pyqtSignal(str)  
    def __init__(self,
                 window_width:int=1500,
                 window_height:int=900):
        super().__init__()
        self.window_width=window_width
        self.window_height=window_height
        self.is_click=False
        self.last_click_time=0

        self.init_ui()
        self.timer=QTimer()
        self.timer.timeout.connect(self.check_button_click)
        self.timer.start(30)

        self.gesture_layer = GestureLayer(self)
        self.gesture_layer.setGeometry(0, 0, self.width(), self.height())
    

    def update_gesture(self, is_click, landmarks):
        self.is_click = is_click
        self.landmarks = landmarks  # 更新主窗口的 landmarks
        self.gesture_layer.update_gesture(landmarks)  # 更新手势层的 landmarks

    def init_ui(self):
        # 基础设置
        self.setGeometry(100, 100, self.window_width, self.window_height)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.black)
        self.setPalette(palette)

        # 主布局
        self.main_layout = QtWidgets.QStackedLayout()
        self.setLayout(self.main_layout)

        # 创建功能部件
        self.func_widget_init()
        # 连接信号
        self.connect_signals()
    
    #负责创建和添加第二层功能部件
    def func_widget_init(self):
        self.func_widgets={
            'main_content':MainContentWidget(),
            'text_reg':TextRegWidget(),
            'speech_reg':SpeechRegWidget(),
            'remote_ctrl':RemoteCtrlWidget(),
            'remote_camera':RemoteCameraWidget(),
            'object_detection':ObjectDetectionWidget()
        }

        self.func_indices={}
        for idx,(name,widget) in enumerate(self.func_widgets.items()):
            self.main_layout.addWidget(widget)
            self.func_indices[name]=idx
    
    #连接信号
    def connect_signals(self):
        #主内容按钮连接
        main_content=self.func_widgets['main_content']
        main_content.buttons["text_reg"].clicked.connect(lambda:self.handle_button("text_reg"))
        main_content.buttons["speech_reg"].clicked.connect(lambda:self.handle_button("speech_reg"))
        main_content.buttons["remote_ctrl"].clicked.connect(lambda:self.handle_button("remote_ctrl"))
        main_content.buttons["remote_camera"].clicked.connect(lambda:self.handle_button("remote_camera"))
        main_content.buttons["object_detection"].clicked.connect(lambda:self.handle_button("object_detection"))

        #返回按钮连接
        for name in ['text_reg','speech_reg','remote_ctrl','remote_camera','object_detection']:
            widget=self.func_widgets[name]
            widget.back_btn.clicked.connect(lambda:self.handle_button("main_content"))
        
        #特殊按钮处理
        self.func_widgets['text_reg'].translate_btn.clicked.connect(lambda:self.handle_button("text_trans"))

    
    def handle_button(self,button_name):
        if button_name in self.func_indices:
            index=self.func_indices[button_name]
            self.main_layout.setCurrentIndex(index)
            #特殊内容处理
            if button_name=='speech_reg':
                self.func_signal.emit(1)
            elif button_name=='remote_camera':
                # self.func_signal.emit(2)
                self.func_widgets['remote_camera'].rcam_need_signal.emit(True)
            elif button_name=='remote_ctrl':
                self.func_signal.emit(3)
            elif button_name=='text_reg':
                print("text_reg")
                self.func_widgets['text_reg'].text_need_signal.emit(True)
            elif button_name=='image_reg':
                self.func_signal.emit(5)
            elif button_name=='object_detection':
                print("object_detection")
            elif button_name=='main_content':
                self.func_widgets['text_reg'].text_need_signal.emit(False)
                self.func_widgets['text_reg'].ppocr_running=False
                self.func_widgets['object_detection'].od_need_signal.emit(False)
                self.func_widgets['remote_camera'].rcam_need_signal.emit(False)
        else:
            if button_name=="text_trans":
                print("翻译按钮按下")
                self.func_widgets['text_reg'].text_need_signal.emit(False)
                self.func_widgets['text_reg'].ppocr_running=True

        self.gesture_layer.raise_()  

    def check_button_click(self):
        if not self.is_click or not self.landmarks or len(self.landmarks) < 18:  # 需要完整手部关键点
            return
        current_time = time.time()
        if current_time - self.last_click_time < 1.5:
            return
        tip = self.landmarks[8]   # 食指尖端
        dip = self.landmarks[7]   # 食指第二关节
        click_vector = (tip[0]-dip[0], tip[1]-dip[1])
        vector_length = (click_vector[0]**2 + click_vector[1]**2)**0.5
        if vector_length < 0.03 or click_vector[1] < 0.02:  # 排除小幅移动和横向移动
            return
        x_pos = int(tip[0] * self.width())
        y_pos = int(tip[1] * self.height())
        buttons = self.get_current_buttons(self.main_layout.currentIndex())
        for btn in buttons:
            if self.is_point_in_button(x_pos, y_pos, btn):
                btn.click()
                self.last_click_time = current_time
                break

    def get_current_buttons(self,c_index):
        """根据当前页面索引获取按钮列表"""
        if c_index == 0:  # 主界面
            return list(self.func_widgets["main_content"].buttons.values())
        elif c_index in range(1, 7):  # 其他功能页面
            content_names = ["text_reg", "speech_reg", "remote_ctrl", "remote_camera", "photo"]
            widget = self.func_widgets[content_names[c_index-1]]
            # 如果是photo界面，返回三个按钮（编辑、保存、返回）
            if content_names[c_index-1]=='photo':
                return [widget.edit_btn, widget.save_btn, widget.back_btn]
            # 其他界面只返回返回按钮
            elif content_names[c_index-1]=='remote_ctrl':
                return [widget.lamp_on_btn,widget.lamp_off_btn,*widget.fan_speed_btns,widget.back_btn]
            elif content_names[c_index-1]=='text_reg':
                return [widget.translate_btn,widget.back_btn]
            else:
                return [widget.back_btn]
        return []
  
    def is_point_in_button(self, x, y, button):
        """改进的按钮点击检测（考虑布局间距）"""
        # 获取按钮在窗口中的实际位置
        btn_pos = button.mapTo(self, QtCore.QPoint(0, 0))
        btn_rect = QtCore.QRect(btn_pos.x(), btn_pos.y(), 
                            button.width(), button.height())
        
        return btn_rect.contains(int(x), int(y))
    
    def is_fist_gesture(self, landmarks):
        """检测是否是握拳手势"""
        if len(landmarks) < 21:
            return False
        wrist = landmarks[0]
        tips = [4, 8, 12, 16, 20] 
        
        distances = []
        for tip in tips:
            tip_pos = landmarks[tip]
            dist = ((tip_pos[0]-wrist[0])**2 + (tip_pos[1]-wrist[1])**2)**0.5
            distances.append(dist)
        avg_dist = sum(distances) / len(distances)
        
        threshold = 0.15  # 这个阈值可能需要根据实际情况调整
        return avg_dist < threshold


if __name__ == '__main__':
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec())
