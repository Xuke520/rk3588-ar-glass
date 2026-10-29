import cv2

# UDP流地址（需替换为发送端IP）
# stream_url = "udp://@192.168.137.26:8554"
# cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)


cap = cv2.VideoCapture("rtmp://192.168.137.26:1096/live/mystream", cv2.CAP_FFMPEG)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("连接中断，检查网络或发送端")
        break
    cv2.imshow('UDP视频流', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




