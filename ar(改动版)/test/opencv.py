
import cv2
 
cap = cv2.VideoCapture(0)  # 打开默认摄像头
 
# 设置分辨率为640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
while True:
    ret, frame = cap.read()  # 读取帧
    if not ret:
        print("无法获取图像")
        break
 
    cv2.imshow('Camera Resolution', frame)  # 显示图像
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
