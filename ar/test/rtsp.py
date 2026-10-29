import cv2
import time

def receive_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    # 设置缓冲区大小（减少延迟）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to grab frame. Reconnecting...")
                time.sleep(1)  # 等待后重试
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue
            
            # 显示帧
            cv2.imshow("RTSP Stream", frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你的RTSP地址
    rtsp_url = "rtsp://192.168.137.184:8554/cam"
    receive_rtsp_stream(rtsp_url)