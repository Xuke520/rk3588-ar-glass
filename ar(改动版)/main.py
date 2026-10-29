import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hand_gesture import Camera,GestureThread
from widget_main import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from medicine_data_init import init_database

if __name__ == "__main__":
    app=QApplication(sys.argv)

    # 初始化药品数据库（首次运行自动填充）
    print("正在初始化药品数据库...")
    init_database()
    print("药品数据库初始化完成")

    camera=Camera(21)

    gesture_thread=GestureThread(camera)
    window=MainWindow()

    # ---- 药物识别：传递 PP-OCR 实例 ----
    medicine_widget = window.func_widgets['medicine']
    medicine_widget.set_ppocr(window.func_widgets['text_reg'].ppocr)

    # 连接手势线程的摄像头画面到药物识别模块
    gesture_thread.image_singal.connect(medicine_widget.update_camera_frame)

    # 药物识别页面激活/停用时，控制手势线程是否发帧
    medicine_widget.medicine_need_signal.connect(
        gesture_thread.send_img_to_medicine,
        Qt.ConnectionType.QueuedConnection
    )

    window.func_widgets['text_reg'].text_need_signal.connect(gesture_thread.send_img_to_text,
                                                             Qt.ConnectionType.QueuedConnection)
    window.func_widgets['photo'].photo_need_signal.connect(gesture_thread.send_img_to_photo,
                                                             Qt.ConnectionType.QueuedConnection)
    gesture_thread.gesture_signal.connect(window.update_gesture)
    gesture_thread.image_singal.connect(window.func_widgets['text_reg'].update_text)
    gesture_thread.image_singal.connect(window.func_widgets['photo'].update_photo)


    gesture_thread.start()  # 先启动线程
    window.show()

    sys.exit(app.exec())

    gesture_thread.stop()
