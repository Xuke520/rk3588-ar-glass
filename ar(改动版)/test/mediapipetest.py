import cv2
import os
import mmap
import threading
import numpy as np
import mediapipe as mp
import math
import time

hand_connections=[
    (0,1),(1,2),(2,3),(3,4),        # 拇指
    (0,5),(5,6),(6,7),(7,8),        # 食指
    (9,10),(10,11),(11,12),   # 中指
    (13,14),(14,15),(15,16), # 无名指
    (0,17),(17,18),(18,19),(19,20), # 小指
    (5,9),(9,13),(13,17)               # 手掌基部
]

def point_distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):
    AB=[x2-x1,y2-y1]
    CD=[x4-x3,y4-y3]

    #计算线段长度
    AB_len=point_distance(x1,y1,x2,y2)+0.0001
    CD_len=point_distance(x3,y3,x4,y4)+0.0001

    cos_theta=(AB[0]*CD[0]+AB[1]*CD[1])/(AB_len*CD_len)
    theta=math.acos(cos_theta)
    return theta

"判断手指状态:"
"大拇指:弯曲或伸直"
"其它:弯曲,伸直或者均不是 "
def judge_finger_bend(landmarks):
    first_bend_ths=math.pi*0.25
    other_bend_ths=math.pi*0.5
    other_straight_ths=math.pi*0.2

    first_is_bend=False
    first_is_straight=False
    second_is_bend=False
    second_is_straight=False
    third_is_bend=False
    third_is_straight=False
    fourth_is_bend=False
    fourth_is_straight=False
    fifth_is_bend=False
    fifth_is_straight=False

    first_angle=compute_angle(landmarks[0][0],landmarks[0][1],
                              landmarks[1][0],landmarks[1][1],
                              landmarks[2][0],landmarks[2][1],
                              landmarks[4][0],landmarks[4][1])
    
    second_angle=compute_angle(landmarks[0][0],landmarks[0][1],
                               landmarks[5][0],landmarks[5][1],
                               landmarks[6][0],landmarks[6][1],
                               landmarks[8][0],landmarks[8][1])
    
    third_angle=compute_angle(landmarks[0][0],landmarks[0][1],
                              landmarks[9][0],landmarks[9][1],
                              landmarks[10][0],landmarks[10][1],
                              landmarks[12][0],landmarks[12][1])
    
    fourth_angle=compute_angle(landmarks[0][0],landmarks[0][1],
                              landmarks[13][0],landmarks[13][1],
                              landmarks[14][0],landmarks[14][1],
                              landmarks[16][0],landmarks[16][1])
    
    fifth_angle=compute_angle(landmarks[0][0],landmarks[0][1],
                              landmarks[17][0],landmarks[17][1],
                              landmarks[18][0],landmarks[18][1],
                              landmarks[20][0],landmarks[20][1])

    if first_angle>first_bend_ths:
        first_is_bend=True
        first_is_straight=False
    else:
        first_is_bend=False
        first_is_straight=True
    
    if second_angle>other_bend_ths:
        second_is_bend=True
    elif second_angle<other_straight_ths:
        second_is_straight=True
    else:
        second_is_bend=False
        second_is_straight=False

    if third_angle>other_bend_ths:
        third_is_bend=True
    elif third_angle<other_straight_ths:
        third_is_straight=True
    else:
        third_is_bend=False
        third_is_straight=False

    if fourth_angle>other_bend_ths:
        fourth_is_bend=True
    elif fourth_angle<other_straight_ths:
        fourth_is_straight=True
    else:
        fourth_is_bend=False
        fourth_is_straight=False

    if fifth_angle>other_bend_ths:
        fifth_is_bend=True
    elif fifth_angle<other_straight_ths:
        fifth_is_straight=True
    else:
        fifth_is_bend=False
        fifth_is_straight=False

    bend_states={'first':first_is_bend,
                 'second':second_is_bend,
                 'third':third_is_bend,
                 'fourth':fourth_is_bend,
                 'fifth':fifth_is_bend}
    straight_states={'first':first_is_straight,
                 'second':second_is_straight,
                 'third':third_is_straight,
                 'fourth':fourth_is_straight,
                 'fifth':fifth_is_straight}

    return bend_states,straight_states




class Camera():
    def __init__(self,
                 cam_id=0,
                 width=640,
                 height=480):
        self.cam_id=cam_id
        self.width=width
        self.height=height


class KalmanFilter():
    def __init__(self,
                 process_noise=0.01,
                 measure_noise=0.5,
                 max_predict_frames=10):
        self.process_noise=process_noise
        self.measure_noise=measure_noise
        self.max_predict_frames=max_predict_frames      #当检测不到手掌时最多预测几帧
        self.kfs=[self.create() for _ in range(21)]
        self.last_filtered_landmarks=None
        self.count=0
        self.hand_type='Right'

    def create(self):
        kf=cv2.KalmanFilter(6,3)
        kf.transitionMatrix=np.array([
            [1,0,0,0.3,0,0,0.5*0.3**2,0,0], 
            [0,1,0,0,0.3,0,0,0.5*0.3**2,0],
            [0,0,1,0,0,0.3,0,0,0.5*0.3**2], 
            [0,0,0,1,0,0,0.3,0,0],   
            [0,0,0,0,1,0,0,0.3,0],  
            [0,0,0,0,0,1,0,0,0.3],   
            [0,0,0,0,0,0,1,0,0],   
            [0,0,0,0,0,0,0,1,0],   
            [0,0,0,0,0,0,0,0,1]     
        ],dtype=np.float32)
        # 观测矩阵（仍只观测位置）
        kf.measurementMatrix=np.array([
            [1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0]
        ],dtype=np.float32)
        # 调整噪声协方差（需要重新调参）
        kf.processNoiseCov=np.eye(9,dtype=np.float32)*self.process_noise
        kf.measurementNoiseCov=np.eye(3,dtype=np.float32)*self.measure_noise
        # 初始状态（位置和速度）
        kf.statePost=np.zeros((9,1),dtype=np.float32)
        # 初始协方差矩阵
        kf.errorCovPost=np.eye(9,dtype=np.float32)
        return kf
    
    def set_transitionMatrix(self,kf,dt):
        kf.transitionMatrix[0,3]=dt
        kf.transitionMatrix[1,4]=dt
        kf.transitionMatrix[2,5]=dt
        kf.transitionMatrix[3,6]=dt
        kf.transitionMatrix[4,7]=dt
        kf.transitionMatrix[5,8]=dt
        kf.transitionMatrix[0,6]=0.5*dt**2
        kf.transitionMatrix[1,7]=0.5*dt**2
        kf.transitionMatrix[2,8]=0.5*dt**2

    def update(self,kf,measurement,dt):
        self.set_transitionMatrix(kf,dt)
        kf.predict()
        measurement=np.array([[measurement.x],[measurement.y],[measurement.z]],dtype=np.float32)
        kf.correct(measurement)

    def predict(self,kf,dt):
        self.set_transitionMatrix(kf,dt)
        kf.predict()

    def get_state(self,kf):
        return (float(kf.statePost[0,0]),float(kf.statePost[1,0]),float(kf.statePost[2,0]))

    def run(self,landmarks):
        filtered_landmarks=None
        is_valid=False

        if landmarks and landmarks.multi_hand_landmarks:
            hand_type=landmarks.multi_handedness[0].classification[0].label
            if hand_type!=self.hand_type:#镜像
                hand=landmarks.multi_hand_landmarks[0]
                self.count=0
                current_hand=[]
                for landmark_idx,landmark in enumerate(hand.landmark):
                    kf=self.kfs[landmark_idx]
                    self.update(kf,landmark,0.3)
                    current_hand.append(self.get_state(kf))
                filtered_landmarks=current_hand
                is_valid=True
                self.last_filtered_landmarks=filtered_landmarks
        # else:#没有检查到手部且预测次数未达上限``
        #     if self.count<self.max_predict_frames:
        #         self.count+=1
        #         if self.last_filtered_landmarks:
        #             predicted_hand=[]
        #             for landmark_idx in range(21):
        #                 kf=self.kfs[landmark_idx]
        #                 self.predict(kf,0.03)
        #                 predicted_hand.append(self.get_state(kf))
        #             filtered_landmarks=predicted_hand
        #             is_valid=True
        #             self.last_filtered_landmarks=filtered_landmarks
        return is_valid,filtered_landmarks


class GestureThread(threading.Thread):
    def __init__(self,
                 camera:Camera,
                 kalmanfilter:KalmanFilter,
                 min_detection_conf=0.75,
                 min_tracking_conf=0.75):
        threading.Thread.__init__(self)
        self.lock=threading.Lock()  # 保护共享数据
        self.camera=camera
        self.landmarks=None
        self.frame=None
        self.running=False
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.kf=kalmanfilter
        self.mp_drawing=mp.solutions.drawing_utils
        
    def run(self):
        cap=cv2.VideoCapture(self.camera.cam_id)
        if not cap.isOpened():
            print("camera open failed\r\n")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.camera.height)
        self.running=True
        while self.running:
            ret1,frame=cap.read()
            if not ret1:
                break
    
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            landmarks=self.hands.process(rgb_frame)
            ret2,filtered_landmarks=self.kf.run(landmarks)
            with self.lock:
                self.frame=frame
                if ret2:
                    self.landmarks=filtered_landmarks
                else:
                    self.landmarks=None
        cap.release()

    def stop(self):
        self.running=False
        self.join()
        self.hands.close()

    def get_landmarks(self):
        with self.lock:
            return self.landmarks,self.frame
    


if __name__ == "__main__":
    camera=Camera(0)
    kf=KalmanFilter()
    gesture_thread=GestureThread(camera,kf)
    gesture_thread.start()
    try:
        while True:
            filtered_landmarks,frame=gesture_thread.get_landmarks()
            if frame is None:
                continue
            h, w=frame.shape[:2]  # 获取画面尺寸
            if filtered_landmarks:
                for landmark in filtered_landmarks:  # 遍历每只手
                    x=int(landmark[0] * w)
                    y=int(landmark[1] * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
                # 绘制连接线
                for start_idx, end_idx in hand_connections:
                    if start_idx < len(filtered_landmarks) and end_idx < len(filtered_landmarks):
                        start_x=int(filtered_landmarks[start_idx][0] * w)
                        start_y=int(filtered_landmarks[start_idx][1] * h)
                        end_x=int(filtered_landmarks[end_idx][0] * w)
                        end_y=int(filtered_landmarks[end_idx][1] * h)
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    gesture_thread.stop()
    cv2.destroyAllWindows()