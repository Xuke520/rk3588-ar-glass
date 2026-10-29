import os
import sys
import cv2
import numpy as np
from PyQt6 import QtCore,QtGui,QtWidgets
from PyQt6.QtWidgets import QApplication,QWidget,QGridLayout,QPushButton,QHBoxLayout,QVBoxLayout,QLabel,QMessageBox,QFileDialog,QLineEdit,QSlider
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
                print("已选择文件: %s" % self.selected_file_path)
   
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
            os.path.join(os.path.dirname(__file__), "fonts/simhei.ttf"),
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf"
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
            #                 print("绘制文本失败: %s, text=%s" % (e, text))
                


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
    # 定义信号，用于与主窗口通信
    robot_control_signal = pyqtSignal(str)  # 发送控制命令
    rcam_need_signal = pyqtSignal(bool)  # 请求视频流
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.connected = False  # 蓝牙连接状态
        self.current_speed = 50  # 当前速度，范围0-100
        self.is_controlling = False  # 是否正在手势控制
        self.current_direction = None  # 当前控制方向
        self.gesture_timer = QTimer()  # 手势控制定时器
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        # 主水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)
        
        # ===== 左侧：视频显示区域 =====
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(15)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                min-width: 640px;
                min-height: 480px;
            }
        """)
        self.video_label.setScaledContents(True)
        
        # 视频控制按钮
        video_control_layout = QHBoxLayout()
        self.start_stream_btn = QPushButton("开始视频流")
        self.stop_stream_btn = QPushButton("停止视频流")
        
        btn_style = """
            QPushButton {
                font-size: 14px;
                border-radius: 8px;
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        for btn in [self.start_stream_btn, self.stop_stream_btn]:
            btn.setStyleSheet(btn_style)
        
        self.stop_stream_btn.setEnabled(False)
        
        video_control_layout.addStretch()
        video_control_layout.addWidget(self.start_stream_btn)
        video_control_layout.addWidget(self.stop_stream_btn)
        video_control_layout.addStretch()
        
        video_layout.addWidget(self.video_label)
        video_layout.addLayout(video_control_layout)
        
        # ===== 右侧：控制区域 =====
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(20)
        control_container.setFixedWidth(400)
        
        # 顶部连接状态
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(15)
        
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px 15px;
                border-radius: 8px;
                background-color: #ff4444;
                color: white;
                min-width: 120px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.connect_btn = QPushButton("连接机器人")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                border-radius: 8px;
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.connect_btn)
        
        # 速度控制
        speed_group = QWidget()
        speed_group.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        speed_layout = QVBoxLayout(speed_group)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(10)
        
        speed_title = QLabel("速度控制")
        speed_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333333;")
        
        speed_slider_layout = QHBoxLayout()
        speed_slider_layout.setSpacing(15)
        
        speed_label = QLabel("速度:")
        speed_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #666666;")
        speed_label.setFixedWidth(60)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -2px 0;
            }
            QSlider::groove:horizontal {
                background-color: #e0e0e0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        
        self.speed_value = QLabel("50%")
        self.speed_value.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        self.speed_value.setFixedWidth(60)
        self.speed_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        speed_slider_layout.addWidget(speed_label)
        speed_slider_layout.addWidget(self.speed_slider)
        speed_slider_layout.addWidget(self.speed_value)
        
        speed_layout.addWidget(speed_title)
        speed_layout.addLayout(speed_slider_layout)
        
        # 方向控制区域（手势控制）
        direction_group = QWidget()
        direction_group.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        direction_layout = QVBoxLayout(direction_group)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.setSpacing(15)
        
        direction_title = QLabel("方向控制")
        direction_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333333;")
        
        # 方向按钮网格
        button_grid = QWidget()
        button_layout = QGridLayout(button_grid)
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(20, 10, 20, 10)
        
        # 创建方向按钮（使用key1图标并旋转）
        self.direction_btns = {}
        
        # 方向配置：(文本, 旋转角度)
        directions = {
            "forward": ("前进", 0),
            "backward": ("后退", 180),
            "left": ("左转", 270),
            "right": ("右转", 90),
            "stop": ("停止", 0)
        }
        
        # 按钮位置布局
        positions = {
            "forward": (0, 1),
            "backward": (2, 1),
            "left": (1, 0),
            "right": (1, 2),
            "stop": (1, 1)
        }
        
        btn_size = 100
        for direction, (text, rotation) in directions.items():
            if direction == "stop":
                btn = create_tool_button(text, "off.png", 80)
            else:
                btn = self.create_rotated_button(text, "key1.png", rotation)
            
            btn.setFixedSize(btn_size, btn_size)
            btn.setEnabled(False)
            
            # 设置手势控制样式
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(76, 175, 80, 0.1);
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: rgba(76, 175, 80, 0.3);
                }
                QPushButton:pressed {
                    background-color: rgba(76, 175, 80, 0.6);
                }
                QPushButton:disabled {
                    background-color: rgba(200, 200, 200, 0.1);
                    border: 2px solid #cccccc;
                }
            """)
            
            self.direction_btns[direction] = btn
            button_layout.addWidget(btn, positions[direction][0], positions[direction][1])
        
        # 手势控制提示
        gesture_hint = QLabel("手势控制提示：按住方向图标控制机器人移动")
        gesture_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gesture_hint.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666666;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """)
        
        direction_layout.addWidget(direction_title)
        direction_layout.addWidget(button_grid)
        direction_layout.addWidget(gesture_hint)
        
        # 机器人状态显示
        self.robot_status = QLabel("机器人状态: 待机")
        self.robot_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.robot_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                background-color: #e8f5e9;
                border-radius: 10px;
                color: #2e7d32;
            }
        """)
        
        # 返回按钮
        self.back_btn = create_tool_button("返回", "back.png", 80)
        self.back_btn.setFixedSize(200, 100)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 2px solid #cccccc;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
                border-color: #999999;
            }
        """)
        
        # 组合右侧布局
        control_layout.addWidget(status_bar)
        control_layout.addWidget(speed_group)
        control_layout.addWidget(direction_group)
        control_layout.addWidget(self.robot_status)
        control_layout.addStretch()
        control_layout.addWidget(self.back_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 组合主布局
        main_layout.addWidget(video_container)
        main_layout.addWidget(control_container)
        
        # 设置整体样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)
    
    def create_rotated_button(self, text, icon_path, rotation):
        """创建带旋转图标的按钮"""
        button = QtWidgets.QPushButton()
        button.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        
        layout = QtWidgets.QVBoxLayout(button)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 创建旋转图标
        icon_label = QtWidgets.QLabel()
        icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # 加载并旋转图标
        pixmap = QtGui.QPixmap("image/%s" % icon_path)
        if not pixmap.isNull():
            transform = QtGui.QTransform().rotate(rotation)
            rotated_pixmap = pixmap.transformed(transform, QtCore.Qt.TransformationMode.SmoothTransformation)
            icon_label.setPixmap(rotated_pixmap.scaled(60, 60, 
                                                     Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        
        # 创建文本标签
        text_label = QtWidgets.QLabel(text)
        text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        
        return button
    
    def connect_signals(self):
        # 连接按钮信号
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.speed_slider.valueChanged.connect(self.update_speed)
        
        # 视频控制信号
        self.start_stream_btn.clicked.connect(self.start_video_stream)
        self.stop_stream_btn.clicked.connect(self.stop_video_stream)
        
        # 机器人控制信号
        self.direction_btns["forward"].clicked.connect(lambda: self.send_command("forward"))
        self.direction_btns["backward"].clicked.connect(lambda: self.send_command("backward"))
        self.direction_btns["left"].clicked.connect(lambda: self.send_command("left"))
        self.direction_btns["right"].clicked.connect(lambda: self.send_command("right"))
        self.direction_btns["stop"].clicked.connect(lambda: self.send_command("stop"))
        
        # 长按事件处理（模拟手势捏住）
        for direction, btn in self.direction_btns.items():
            btn.pressed.connect(lambda checked, d=direction: self.start_continuous_control(d))
            btn.released.connect(self.stop_continuous_control)
        
        # 手势控制定时器
        self.gesture_timer.timeout.connect(self.send_continuous_command)
    
    def toggle_connection(self):
        if not self.connected:
            # 模拟连接机器人
            self.connected = True
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    background-color: #4CAF50;
                    color: white;
                }
            """)
            self.connect_btn.setText("断开连接")
            self.robot_status.setText("机器人状态: 就绪")
            
            # 启用控制按钮
            for btn in self.direction_btns.values():
                btn.setEnabled(True)
            
            # 启用视频流按钮
            self.start_stream_btn.setEnabled(True)
            
            print("已连接到负压爬壁机器人")
        else:
            # 断开连接
            self.connected = False
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    background-color: #ff4444;
                    color: white;
                }
            """)
            self.connect_btn.setText("连接机器人")
            self.robot_status.setText("机器人状态: 待机")
            
            # 停止手势控制
            self.stop_continuous_control()
            
            # 禁用控制按钮
            for btn in self.direction_btns.values():
                btn.setEnabled(False)
            
            # 停止视频流
            self.stop_video_stream()
            self.start_stream_btn.setEnabled(False)
            
            print("已断开与机器人的连接")
    
    def update_speed(self, value):
        self.current_speed = value
        self.speed_value.setText("%d%%" % value)
        print("机器人速度设置为: %d%%" % value)
    
    def start_video_stream(self):
        self.rcam_need_signal.emit(True)
        self.start_stream_btn.setEnabled(False)
        self.stop_stream_btn.setEnabled(True)
        self.robot_status.setText("机器人状态: 视频流已启动")
        print("已启动机器人视频流")
    
    def stop_video_stream(self):
        self.rcam_need_signal.emit(False)
        self.start_stream_btn.setEnabled(True)
        self.stop_stream_btn.setEnabled(False)
        # 清空视频显示
        self.video_label.clear()
        self.robot_status.setText("机器人状态: 视频流已停止")
        print("已停止机器人视频流")
    
    def update_frame(self, img):
        """更新视频帧"""
        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img.scaled(640, 480, 
                                                                           Qt.AspectRatioMode.KeepAspectRatio,
                                                                           Qt.TransformationMode.SmoothTransformation)))
    
    def send_command(self, command):
        if self.connected:
            # 发送控制命令
            self.robot_control_signal.emit(command)
            
            # 更新状态显示
            status_text = {
                "forward": "前进中",
                "backward": "后退中", 
                "left": "左转中",
                "right": "右转中",
                "stop": "已停止"
            }
            
            self.robot_status.setText("机器人状态: %s (速度: %d%%)" % (status_text.get(command, command), self.current_speed))
            print("发送命令: %s (速度: %d%%)" % (command, self.current_speed))
        else:
            self.robot_status.setText("错误: 未连接机器人")
    
    def start_continuous_control(self, direction):
        """开始连续控制（模拟手势捏住）"""
        if self.connected and direction != "stop":
            self.is_controlling = True
            self.current_direction = direction
            self.send_command(direction)
            # 启动定时器，每100ms发送一次命令
            self.gesture_timer.start(100)
            print("开始连续控制: %s" % direction)
    
    def send_continuous_command(self):
        """发送连续控制命令"""
        if self.is_controlling and self.current_direction and self.connected:
            self.robot_control_signal.emit(self.current_direction)
            print("连续控制命令: %s (速度: %d%%)" % (self.current_direction, self.current_speed))
    
    def stop_continuous_control(self):
        """停止连续控制（模拟手势松开）"""
        if self.is_controlling:
            self.is_controlling = False
            self.current_direction = None
            self.gesture_timer.stop()
            self.send_command("stop")
            print("停止连续控制")
    
    def detect_gesture(self, frame):
        """从视频帧中检测手势"""
        try:
            # 这里可以添加实际的手势识别逻辑
            # 目前使用模拟的手势检测
            import random
            
            # 模拟手势检测结果
            gesture_prob = random.random()
            
            if gesture_prob < 0.1:
                # 检测到前进手势
                return "forward"
            elif gesture_prob < 0.2:
                # 检测到后退手势
                return "backward"
            elif gesture_prob < 0.3:
                # 检测到左转手势
                return "left"
            elif gesture_prob < 0.4:
                # 检测到右转手势
                return "right"
            elif gesture_prob < 0.45:
                # 检测到停止手势
                return "stop"
            else:
                # 没有检测到手势
                return None
        except Exception as e:
            print(f"手势检测失败: {e}")
            return None
    
    def process_video_frame(self, frame):
        """处理视频帧，检测手势并控制机器人"""
        # 更新视频显示
        self.update_frame(frame)
        
        # 检测手势
        gesture = self.detect_gesture(frame)
        
        if gesture:
            if gesture == "stop":
                self.stop_continuous_control()
            elif gesture in ["forward", "backward", "left", "right"]:
                if not self.is_controlling or self.current_direction != gesture:
                    self.stop_continuous_control()
                    self.start_continuous_control(gesture)
            print(f"检测到手势: {gesture}")
    
    def update_frame(self, img):
        """更新视频帧"""
        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img.scaled(640, 480, 
                                                                           Qt.AspectRatioMode.KeepAspectRatio,
                                                                           Qt.TransformationMode.SmoothTransformation)))



        
             
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
    video_frame_signal = pyqtSignal(object)  # 发送视频帧
    vehicle_data_received = pyqtSignal(dict)  # 接收车辆数据
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.vehicle_data_timer = QTimer()
        self.is_streaming = False
        self.default_url = 'http://192.168.1.9:8080/video'
        # 车辆数据初始化
        self.alert_items = {
            "obstacle": {"label": "前方障碍物", "value": "无", "status": "normal"},
            "latency": {"label": "总延迟", "value": "0ms", "status": "normal"},
            "connection": {"label": "连接状态", "value": "正常", "status": "normal"}
        }
        self.alert_labels = {}
        self.battery_label = None
        self.temp_label = None
        self.humidity_label = None
        self.object_detection_enabled = False  # 是否启用目标检测
        self.yolo = None  # YOLOv5模型实例
        self.init_ui()
        self.setup_timer()
        
        # 创建信号用于接收实时数据
        self.vehicle_data_received = pyqtSignal(dict)
    
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
        
        # 目标检测控制按钮
        od_layout = QHBoxLayout()
        od_layout.setSpacing(10)
        self.od_toggle_btn = QPushButton('启用目标检测', self)
        self.od_toggle_btn.setFixedHeight(40)
        self.od_toggle_btn.setEnabled(False)
        self.od_toggle_btn.clicked.connect(self.toggle_object_detection)
        od_layout.addWidget(self.od_toggle_btn)
        od_layout.addStretch()
        left_layout.addLayout(od_layout)

        main_layout.addLayout(left_layout, stretch=1)

        # 右侧信息显示区域
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(30)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_container.setFixedWidth(300)
        right_container.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 0.9);
                border-radius: 15px;
                border: 2px solid #4CAF50;
            }
        """)
        
        # 标题
        title_label = QLabel("车辆监控")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(title_label)
        
        # 警报信息区域
        alert_group = QWidget()
        alert_layout = QVBoxLayout(alert_group)
        alert_layout.setSpacing(15)
        alert_layout.setContentsMargins(0, 0, 0, 0)
        
        alert_title = QLabel("警报信息")
        alert_title.setStyleSheet("""
            QLabel {
                color: #FF9800;
                font-size: 18px;
                font-weight: bold;
                border-bottom: 2px solid #FF9800;
                padding-bottom: 8px;
            }
        """)
        alert_layout.addWidget(alert_title)
        
        # 告警项（已在__init__中初始化）
        for key, info in self.alert_items.items():
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setSpacing(10)
            item_layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(f"{info['label']}:")
            label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 14px;
                    min-width: 80px;
                }
            """)
            item_layout.addWidget(label)
            
            value_label = QLabel(info['value'])
            value_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            item_layout.addWidget(value_label)
            
            status_icon = QLabel()
            status_icon.setFixedSize(16, 16)
            status_icon.setStyleSheet("""
                QLabel {
                    background-color: #4CAF50;
                    border-radius: 8px;
                }
            """)
            item_layout.addWidget(status_icon)
            item_layout.addStretch()
            
            alert_layout.addWidget(item_widget)
            self.alert_labels[key] = (value_label, status_icon)
        
        right_layout.addWidget(alert_group)
        
        # 车体信息区域
        vehicle_group = QWidget()
        vehicle_layout = QVBoxLayout(vehicle_group)
        vehicle_layout.setSpacing(15)
        vehicle_layout.setContentsMargins(0, 0, 0, 0)
        
        vehicle_title = QLabel("车体信息")
        vehicle_title.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-size: 18px;
                font-weight: bold;
                border-bottom: 2px solid #2196F3;
                padding-bottom: 8px;
            }
        """)
        vehicle_layout.addWidget(vehicle_title)
        
        # 电池电压
        self.battery_label = QLabel("电池电压: 12.6V")
        self.battery_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        vehicle_layout.addWidget(self.battery_label)
        
        # 环境温度
        self.temp_label = QLabel("环境温度: 25°C")
        self.temp_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        vehicle_layout.addWidget(self.temp_label)
        
        # 环境湿度
        self.humidity_label = QLabel("环境湿度: 45%")
        self.humidity_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        vehicle_layout.addWidget(self.humidity_label)
        
        right_layout.addWidget(vehicle_group)
        
        # 返回按钮
        self.back_btn = create_tool_button("返回", 'back.png', 150)
        self.back_btn.setFixedSize(250, 200)
        self.back_btn.clicked.connect(self.stop_streaming)
        right_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 将右侧区域添加到主布局
        main_layout.addWidget(right_container, stretch=0)
        
        # 连接信号
        # 移除了rcam_need_signal相关代码，直接通过方法调用控制串流
    
    def setup_timer(self):
        """设置定时器用于更新视频帧"""
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(33)  # 约30帧/秒
        
        # 车辆数据更新定时器
        self.vehicle_data_timer = QTimer()
        self.vehicle_data_timer.timeout.connect(self.simulate_vehicle_data)
        self.vehicle_data_timer.setInterval(1000)  # 每秒更新一次
        
        # 连接实时数据信号
        self.vehicle_data_received.connect(self.receive_vehicle_data)
    
    def init_object_detection(self):
        """初始化目标检测模型"""
        if not self.object_detection_enabled and self.yolo is None:
            try:
                self.yolo = Yolov5RKNN()
                self.object_detection_enabled = True
                print("目标检测模型已初始化")
            except Exception as e:
                print(f"初始化目标检测模型失败: {e}")
                self.object_detection_enabled = False
    
    def destroy_object_detection(self):
        """销毁目标检测模型"""
        if self.yolo is not None:
            # 这里可以添加模型销毁的逻辑
            self.yolo = None
            self.object_detection_enabled = False
            print("目标检测模型已销毁")
    
    def toggle_object_detection(self):
        """切换目标检测的启用/禁用状态"""
        if not self.is_streaming:
            print("请先启动视频流")
            return
            
        if self.object_detection_enabled:
            # 禁用目标检测
            self.destroy_object_detection()
            self.od_toggle_btn.setText('启用目标检测')
            self.od_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
        else:
            # 启用目标检测
            self.init_object_detection()
            if self.object_detection_enabled:
                self.od_toggle_btn.setText('禁用目标检测')
                self.od_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #F44336;
                        color: white;
                        border: none;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                self.od_toggle_btn.setText('启用目标检测（失败）')
    
    # 移除了on_streaming_toggle方法，直接通过start_streaming和stop_streaming方法控制串流
    
    def on_connect_clicked(self):
        url = self.stream_url_input.text().strip()
        self.start_streaming(url)

    def start_streaming(self, url: str = None):
        """开始远程串流"""
        if self.is_streaming:
            self.stop_streaming()

        if not url:
            url = self.stream_url_input.text().strip()
            if not url:
                self.show_error_message("请先输入有效的流地址")
                return

        self.stream_url_input.setText(url)
        # 清空之前的显示
        self.image_label.clear()
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
                print("错误：无法连接到远程摄像头 %s" % url)
                self.show_error_message("无法连接到远程摄像头: %s\n请确认使用 http 协议并在浏览器/ffplay 中可访问。" % url)
                return

            self.is_streaming = True
            self.timer.start()
            self.vehicle_data_timer.start()  # 启动车辆数据更新定时器
            
            # 初始化目标检测模型
            self.init_object_detection()
            
            # 启用目标检测控制按钮
            self.od_toggle_btn.setEnabled(True)
            if self.object_detection_enabled:
                self.od_toggle_btn.setText('禁用目标检测')
                self.od_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #F44336;
                        color: white;
                        border: none;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                self.od_toggle_btn.setText('启用目标检测')
                self.od_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2196F3;
                        color: white;
                        border: none;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #1976D2;
                    }
                """)
            
            print("远程串流已启动: %s" % url)

        except Exception as e:
            print("启动远程串流失败: %s" % str(e))
            self.show_error_message("启动失败: %s" % str(e))
    
    def stop_streaming(self):
        """停止远程串流"""
        if self.timer.isActive():
            self.timer.stop()
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        self.is_streaming = False
        
        # 停止车辆数据更新定时器
        if self.vehicle_data_timer.isActive():
            self.vehicle_data_timer.stop()
        
        # 销毁目标检测模型
        self.destroy_object_detection()
        
        # 禁用目标检测控制按钮
        self.od_toggle_btn.setEnabled(False)
        self.od_toggle_btn.setText('启用目标检测')
        
        # 清空图片显示
        self.image_label.clear()
        print("远程串流已停止")
    
    def update_frame(self):
        """更新视频帧显示"""
        if not self.is_streaming or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # 如果启用了目标检测，执行检测
                if self.object_detection_enabled and self.yolo is not None:
                    try:
                        # 执行目标检测（只识别人）
                        frame = self.yolo.infer(frame.copy())
                    except Exception as e:
                        print(f"目标检测失败: {e}")
                
                # 发送视频帧信号给其他模块
                self.video_frame_signal.emit(frame)
                
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
                # 读取失败时不停止串流，继续尝试
                print("视频帧读取失败，继续尝试...")
                
        except Exception as e:
            print("更新视频帧失败: %s" % str(e))
            # 不停止串流，继续尝试
    
    def show_error_message(self, message):
        """显示错误消息"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("错误")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def update_vehicle_info(self, obstacle=None, latency=None, connection=None, 
                          battery=None, temperature=None, humidity=None):
        """更新车辆信息和告警状态"""
        # 更新警报信息
        if obstacle is not None:
            self.alert_items["obstacle"]["value"] = obstacle
            value_label, status_icon = self.alert_labels["obstacle"]
            value_label.setText(obstacle)
            # 根据障碍物距离判断状态
            try:
                if obstacle != "无":
                    distance = float(obstacle.replace("cm", ""))
                    if distance < 50:
                        self.alert_items["obstacle"]["status"] = "danger"
                        value_label.setStyleSheet("""
                            QLabel {
                                color: #F44336;
                                font-size: 14px;
                                font-weight: bold;
                            }
                        """)
                        status_icon.setStyleSheet("""
                            QLabel {
                                background-color: #F44336;
                                border-radius: 8px;
                            }
                        """)
                    elif distance < 100:
                        self.alert_items["obstacle"]["status"] = "warning"
                        value_label.setStyleSheet("""
                            QLabel {
                                color: #FF9800;
                                font-size: 14px;
                                font-weight: bold;
                            }
                        """)
                        status_icon.setStyleSheet("""
                            QLabel {
                                background-color: #FF9800;
                                border-radius: 8px;
                            }
                        """)
                    else:
                        self.alert_items["obstacle"]["status"] = "normal"
                        value_label.setStyleSheet("""
                            QLabel {
                                color: #4CAF50;
                                font-size: 14px;
                                font-weight: bold;
                            }
                        """)
                        status_icon.setStyleSheet("""
                            QLabel {
                                background-color: #4CAF50;
                                border-radius: 8px;
                            }
                        """)
                else:
                    self.alert_items["obstacle"]["status"] = "normal"
                    value_label.setStyleSheet("""
                        QLabel {
                            color: #4CAF50;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    """)
                    status_icon.setStyleSheet("""
                        QLabel {
                            background-color: #4CAF50;
                            border-radius: 8px;
                        }
                    """)
            except ValueError:
                pass
        
        if latency is not None:
            self.alert_items["latency"]["value"] = latency
            value_label, status_icon = self.alert_labels["latency"]
            value_label.setText(latency)
            # 根据网络延迟判断状态
            try:
                delay = float(latency.replace("ms", ""))
                if delay > 80:
                    self.alert_items["latency"]["status"] = "warning"
                    value_label.setStyleSheet("""
                        QLabel {
                            color: #FF9800;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    """)
                    status_icon.setStyleSheet("""
                        QLabel {
                            background-color: #FF9800;
                            border-radius: 8px;
                        }
                    """)
                else:
                    self.alert_items["latency"]["status"] = "normal"
                    value_label.setStyleSheet("""
                        QLabel {
                            color: #4CAF50;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    """)
                    status_icon.setStyleSheet("""
                        QLabel {
                            background-color: #4CAF50;
                            border-radius: 8px;
                        }
                    """)
            except ValueError:
                pass
        
        if connection is not None:
            self.alert_items["connection"]["value"] = connection
            value_label, status_icon = self.alert_labels["connection"]
            value_label.setText(connection)
            if connection != "正常":
                self.alert_items["connection"]["status"] = "danger"
                value_label.setStyleSheet("""
                    QLabel {
                        color: #F44336;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)
                status_icon.setStyleSheet("""
                    QLabel {
                        background-color: #F44336;
                        border-radius: 8px;
                    }
                """)
            else:
                self.alert_items["connection"]["status"] = "normal"
                value_label.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)
                status_icon.setStyleSheet("""
                    QLabel {
                        background-color: #4CAF50;
                        border-radius: 8px;
                    }
                """)
        
        # 更新车体信息
        if battery is not None:
            self.battery_label.setText(f"电池电压: {battery}")
        if temperature is not None:
            self.temp_label.setText(f"环境温度: {temperature}")
        if humidity is not None:
            self.humidity_label.setText(f"环境湿度: {humidity}")
    
    def simulate_vehicle_data(self):
        """模拟车辆数据更新（用于测试）"""
        import random
        
        # 模拟障碍物检测
        obstacle = "无" if random.random() > 0.1 else f"{random.randint(50, 200)}cm"
        
        # 模拟网络延迟（限制在100ms以内）
        latency = random.randint(0, 100)
        latency_str = f"{latency}ms"
        
        # 模拟连接状态
        connection = "正常" if random.random() > 0.05 else "不稳定"
        
        # 模拟电池电压
        battery_voltage = 12.6 + random.uniform(-0.5, 0.3)
        battery_str = f"{battery_voltage:.1f}V"
        
        # 模拟环境温度
        temperature = 25 + random.uniform(-5, 10)
        temp_str = f"{temperature:.0f}°C"
        
        # 模拟环境湿度
        humidity = 45 + random.uniform(-10, 10)
        humidity_str = f"{humidity:.0f}%"
        
        self.update_vehicle_info(
            obstacle=obstacle,
            latency=latency_str,
            connection=connection,
            battery=battery_str,
            temperature=temp_str,
            humidity=humidity_str
        )
    
    def receive_vehicle_data(self, data):
        """接收来自机器人的实时数据"""
        try:
            # 解析数据
            obstacle = data.get("obstacle", None)
            latency = data.get("latency", None)
            connection = data.get("connection", None)
            battery = data.get("battery", None)
            temperature = data.get("temperature", None)
            humidity = data.get("humidity", None)
            
            # 更新UI
            self.update_vehicle_info(
                obstacle=obstacle,
                latency=latency,
                connection=connection,
                battery=battery,
                temperature=temperature,
                humidity=humidity
            )
            
            return True
        except Exception as e:
            print(f"解析车辆数据失败: {e}")
            return False
    
    def closeEvent(self, event):
        """确保在关闭窗口时释放资源"""
        self.stop_streaming()
        # 停止车辆数据更新定时器
        self.vehicle_data_timer.stop()
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
            print("启动目标检测失败: %s" % str(e))
            self.show_error_message("启动失败: %s" % str(e))

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
            file_path = "photo_%s.png" % timestamp
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
            self.msg_label.setText("✓ 图片已保存\n%s" % file_path)
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
    pixmap = QtGui.QPixmap("image/%s" % icon_path)
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

        # 安全系数更新定时器
        self.safety_timer = QTimer()
        self.safety_timer.timeout.connect(self.simulate_safety_factor)
        self.safety_timer.start(2000)  # 每2秒更新一次
        self.current_safety_factor = 100

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

        # 创建主容器布局
        self.main_container = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setLayout(self.main_layout)
        
        # 创建顶部状态栏
        self.create_status_bar()
        
        # 创建功能区域布局
        self.function_layout = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.function_layout, stretch=1)

        # 创建功能部件
        self.func_widget_init()
        # 连接信号
        self.connect_signals()
    
    def create_status_bar(self):
        """创建顶部状态栏，显示安全系数等信息"""
        self.status_bar = QtWidgets.QWidget()
        self.status_bar.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 180);
                border-bottom: 2px solid #4CAF50;
            }
        """)
        self.status_bar.setFixedHeight(60)
        
        status_layout = QtWidgets.QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(20, 0, 20, 0)
        status_layout.setSpacing(20)
        
        # 左侧：空白占位
        status_layout.addStretch()
        
        # 右侧：安全系数显示
        safety_container = QtWidgets.QWidget()
        safety_layout = QtWidgets.QHBoxLayout(safety_container)
        safety_layout.setContentsMargins(0, 0, 0, 0)
        safety_layout.setSpacing(10)
        
        safety_icon = QtWidgets.QLabel()
        safety_icon.setPixmap(QtGui.QPixmap("image/shield.png").scaled(40, 40, 
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation))
        safety_layout.addWidget(safety_icon)
        
        safety_label = QtWidgets.QLabel("安全系数:")
        safety_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        safety_layout.addWidget(safety_label)
        
        self.safety_value_label = QtWidgets.QLabel("100%")
        self.safety_value_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 28px;
                font-weight: bold;
            }
        """)
        safety_layout.addWidget(self.safety_value_label)
        
        status_layout.addWidget(safety_container)
        
        self.main_layout.addWidget(self.status_bar)
    
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
            self.function_layout.addWidget(widget)
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
        
        # 远程控制信号连接
        self.func_widgets['remote_ctrl'].robot_control_signal.connect(self.handle_robot_control)
        self.func_widgets['remote_ctrl'].rcam_need_signal.connect(self.handle_remote_camera_need)
        
        # 视频帧信号连接：将RemoteCameraWidget的视频帧发送给RemoteCtrlWidget
        self.func_widgets['remote_camera'].video_frame_signal.connect(
            self.func_widgets['remote_ctrl'].update_frame
        )

        #返回按钮连接
        for name in ['text_reg','speech_reg','remote_ctrl','remote_camera','object_detection']:
            widget=self.func_widgets[name]
            widget.back_btn.clicked.connect(lambda:self.handle_button("main_content"))
        
        #特殊按钮处理
        self.func_widgets['text_reg'].translate_btn.clicked.connect(lambda:self.handle_button("text_trans"))

    
    def handle_button(self,button_name):
        if button_name in self.func_indices:
            index=self.func_indices[button_name]
            self.function_layout.setCurrentIndex(index)
            #特殊内容处理
            if button_name=='speech_reg':
                self.func_signal.emit(1)
            elif button_name=='remote_camera':
                # self.func_signal.emit(2)
                # 不再使用rcam_need_signal，直接调用start_streaming方法
                self.func_widgets['remote_camera'].start_streaming()
            elif button_name=='remote_ctrl':
                self.func_signal.emit(3)
            elif button_name=='text_reg':
                print("text_reg")
                self.func_widgets['text_reg'].text_need_signal.emit(True)
            elif button_name=='object_detection':
                print("object_detection")
            elif button_name=='main_content':
                self.func_widgets['text_reg'].text_need_signal.emit(False)
                self.func_widgets['text_reg'].ppocr_running=False
                self.func_widgets['object_detection'].od_need_signal.emit(False)
                # 不再使用rcam_need_signal，直接调用stop_streaming方法
                self.func_widgets['remote_camera'].stop_streaming()
        else:
            if button_name=="text_trans":
                print("翻译按钮按下")
                self.func_widgets['text_reg'].text_need_signal.emit(False)
                self.func_widgets['text_reg'].ppocr_running=True

        self.gesture_layer.raise_()
    
    def handle_robot_control(self, command):
        """处理机器人控制命令"""
        print("处理机器人命令: %s" % command)
        # 这里需要添加实际的蓝牙通信代码
        # 例如：通过蓝牙串口发送命令到机器人
        # 可以使用pybluez或pyserial库实现
    
    def handle_remote_camera_need(self, need):
        """处理远程摄像头需求"""
        if need:
            print("远程控制模块请求视频流")
            # 启动远程串流
            self.func_widgets['remote_camera'].start_streaming()
        else:
            print("远程控制模块停止视频流")
            # 停止远程串流
            self.func_widgets['remote_camera'].stop_streaming()
    
    def update_safety_factor(self, factor):
        """更新安全系数显示"""
        # 确保系数在0-100之间
        factor = max(0, min(100, factor))
        self.safety_value_label.setText("%d%%" % factor)
        
        # 根据安全系数改变颜色
        if factor >= 80:
            color = "#4CAF50"  # 绿色
        elif factor >= 50:
            color = "#FF9800"  # 橙色
        else:
            color = "#F44336"  # 红色
            
        # 使用普通字符串格式化避免大括号冲突
        style_sheet = """
            QLabel {
                color: %s;
                font-size: 28px;
                font-weight: bold;
            }
        """
        self.safety_value_label.setStyleSheet(style_sheet % color)
        
        print("安全系数已更新为: %d%%" % factor)
    
    def simulate_safety_factor(self):
        """模拟安全系数的变化"""
        # 模拟安全系数的随机波动
        import random
        change = random.randint(-5, 5)
        self.current_safety_factor += change
        
        # 确保安全系数在0-100之间
        self.current_safety_factor = max(0, min(100, self.current_safety_factor))
        
        # 更新显示
        self.update_safety_factor(self.current_safety_factor)

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
        elif c_index in range(1, 6):  # 其他功能页面（1-5对应5个功能）
            content_names = ["text_reg", "speech_reg", "remote_ctrl", "remote_camera", "object_detection"]
            widget = self.func_widgets[content_names[c_index-1]]
            # 不同界面返回不同按钮
            if content_names[c_index-1]=='text_reg':
                return [widget.translate_btn,widget.back_btn]
            elif content_names[c_index-1]=='remote_ctrl':
                return [widget.back_btn, widget.connect_btn, widget.start_stream_btn, widget.stop_stream_btn]
            elif content_names[c_index-1]=='remote_camera':
                return [widget.back_btn, widget.connect_btn, widget.disconnect_btn]
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
