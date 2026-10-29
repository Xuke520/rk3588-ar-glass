import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hand_gesture import Camera,GestureThread
from widget_main import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

if __name__ == "__main__":
    app=QApplication(sys.argv)
    camera=Camera(21)

    gesture_thread=GestureThread(camera)
    window=MainWindow()
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
