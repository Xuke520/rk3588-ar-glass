import cv2

def test_camera():
    # 打开摄像头（0表示默认摄像头）
    # cap = cv2.VideoCapture('https://10.30.236.200:8080')
    cap = cv2.VideoCapture('http://10.30.236.200:8080/video')
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出，按 '空格' 键拍照")
    
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法获取图像")
            break
        
        # 显示图像
        cv2.imshow('Camera Test', frame)
        
        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # 按q退出
            break
        elif key == ord(' '):  # 按空格拍照
            cv2.imwrite('photo.jpg', frame)
            print("已保存照片：photo.jpg")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_camera()