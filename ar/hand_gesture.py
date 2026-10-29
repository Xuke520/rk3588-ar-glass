import cv2
import math
import time
import numpy as np
import mediapipe as mp
from PyQt6.QtCore import pyqtSignal,QThread,QSharedMemory

hand_connections=[
    (0,1),(1,2),(2,3),(3,4),    # 拇指
    (0,5),(5,6),(6,7),(7,8),    # 食指
    (9,10),(10,11),(11,12),     # 中指
    (13,14),(14,15),(15,16),    # 无名指
    (0,17),(17,18),(18,19),(19,20), # 小指
    (5,9),(9,13),(13,17)         # 手掌基部
]

class Camera:
    def __init__(self,
                 cam_id:int=0,
                 width:int=640,
                 height:int=480):
        self.cam_id=cam_id
        self.width=width
        self.height=height

class KalmanFilter:
    def __init__(self,
                 process_noise=0.01,
                 measure_noise=0.5,
                 max_predict_frames=3):
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

    '''
    没有检测到手以后会预测几帧
    若一直没有手则返回空列表
    '''
    def run(self,landmarks):
        filtered_landmarks=[]
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
                self.last_filtered_landmarks=filtered_landmarks
        else:#没有检查到手部且预测次数未达上限``
            if self.count<self.max_predict_frames:
                self.count+=1
                if self.last_filtered_landmarks:
                    predicted_hand=[]
                    for landmark_idx in range(21):
                        kf=self.kfs[landmark_idx]
                        self.predict(kf,0.03)
                        predicted_hand.append(self.get_state(kf))
                    filtered_landmarks=predicted_hand
                    self.last_filtered_landmarks=filtered_landmarks
        return filtered_landmarks



def point_distance(x1,y1,x2,y2):
        return math.sqrt((x2-x1)**2+(y2-y1)**2)

#返回弧度
def compute_angle(x1,y1,x2,y2,x3,y3,x4,y4):
    AB=np.array([x2-x1,y2-y1])
    CD=np.array([x4-x3,y4-y3])
    AB_len=np.linalg.norm(AB)+0.0000001
    CD_len=np.linalg.norm(CD)+0.0000001
    cos_theta=np.dot(AB,CD)/(AB_len*CD_len)
    return np.arccos(cos_theta)

"判断手指状态:"
"大拇指:弯曲或伸直"
"其它:弯曲,伸直或者均不是 "
def get_finger_state(landmarks):
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

class MouseController:
    def __init__(self,
                smoothing:float=0.8,
                cool_time:float=0.2,
                click_threshold:int=10):
        self.smoothing=smoothing
        self.is_click=False
        self.click_count=0
        self.click_threshold=click_threshold
        self.cool_time=max(0,cool_time)
        self.last_click_time=0

    def judge_click(self,landmarks,angle_range=(0.2*math.pi,0.8*math.pi)):
        angle_56_78=compute_angle(landmarks[5][0],landmarks[5][1],
                                landmarks[6][0],landmarks[6][1],
                                landmarks[7][0],landmarks[7][1],
                                landmarks[8][0],landmarks[8][1])
        if not (angle_range[0] < angle_56_78< angle_range[1]):
            return False
        else:
            dis_48=point_distance(landmarks[4][0],landmarks[4][1],
                                landmarks[8][0],landmarks[8][1])
            dis_26=point_distance(landmarks[2][0],landmarks[2][1],
                                landmarks[6][0],landmarks[6][1])
            dis_46=point_distance(landmarks[4][0],landmarks[4][1],
                                landmarks[6][0],landmarks[6][1])
            return dis_48<dis_26 and dis_48<dis_46 and dis_48<0.08

    
    
    def run(self,landmarks)->bool:
        current_time=time.time()
        if current_time-self.last_click_time<self.cool_time:
            self.click_count=0
            self.is_click=False
            return False
        
        if landmarks and self.judge_click(landmarks):
            self.click_count+=1
            if self.click_count>=self.click_threshold and not self.is_click:
                self.is_click=True
                self.last_click_time=current_time
        else:
            self.click_count=max(0,self.click_count-2)
            self.is_click=False
        return self.is_click


class ImageSharedBuf:
    def __init__(self,shape):
        self.shape=shape
        self.dtype=np.uint8
        self.nbytes=np.prod(shape)*np.dtype(self.dtype).itemsize
        self.array=None  # 直接映射共享内存的数组视图
        
    def attach_shm(self,shm: QSharedMemory)->bool:
        #附加到共享内存并创建数组视图
        if not shm.isAttached() and not shm.attach():
            return False
        
        void_ptr=shm.data()
        if not void_ptr:
            return False
            
        #创建直接映射共享内存的numpy数组视图
        self.array=np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=void_ptr,
            order='C'
        )
        return True
        
    def detach_shm(self):
        #分离共享内存
        self.array=None
        
    def read_from_shm(self,shm):
        #更新共享内存引用(零拷贝)
        return self.attach_shm(shm)
        
    def write_to_shm(self,shm,image):
        #写入数据到共享内存(仅当需要转换时才拷贝)
        if not self.attach_shm(shm):
            return False
        if image.shape!=self.shape or image.dtype!=self.dtype:
            return False
        if getattr(image.base,'data',None)==shm.data():
            return True
        np.copyto(self.array,image)
        return True
    


class GestureThread(QThread):
    gesture_signal=pyqtSignal(bool,list)#手势检测结果信号
    image_singal=pyqtSignal(np.ndarray)
    def __init__(self,
                 camera:Camera,
                 min_detection_conf=0.75,
                 min_tracking_conf=0.75):
        super().__init__()
        self.isRunning=False
        self.camera=camera
        self.kf=KalmanFilter()
        self.landmarks=None
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mouse_ctrl=MouseController()

        self.text_img_need=False
        self.photo_img_need=False
        

  
    def run(self):
        cap=cv2.VideoCapture(self.camera.cam_id)
        if not cap.isOpened():
            print("camera open failed\r\n")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.camera.height)
        self.isRunning=True
        while self.isRunning:
            ret1,frame=cap.read()
            if not ret1:
                print("camera read err")
                break
            # 水平翻转，修正镜像问题
            frame = cv2.flip(frame, 1)
            if self.text_img_need or self.photo_img_need:
                self.image_singal.emit(frame)
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            landmarks=self.hands.process(rgb_frame)
            filtered_landmarks=self.kf.run(landmarks)
            is_click=self.mouse_ctrl.run(filtered_landmarks)
            self.gesture_signal.emit(is_click,filtered_landmarks)  #发送信号
            
        cap.release()

    def send_img_to_text(self,need_send):
        self.text_img_need=need_send
        print("text:",need_send)
    
    def send_img_to_photo(self,need_send):
        self.photo_img_need=need_send
        print("photo:",need_send)
    
    def send_img_to_object_detection(self,need_send):
        self.photo_img_need=need_send
        print("object_detection:",need_send)

    def stop(self):
        self.isRunning=False
        self.hands.close()
        self.wait()

















































# class GestureThread(QThread):
#     # shm_send_finish=pyqtSignal()#共享图像准备就绪信号
#     gesture_signal=pyqtSignal(bool,list)#手势检测结果信号
#     image_singal=pyqtSignal(np.ndarray)
#     def __init__(self,
#                  camera:Camera,
#                  min_detection_conf=0.75,
#                  min_tracking_conf=0.75):
#         super().__init__()
#         self.isRunning=False
#         self.camera=camera
#         self.kf=KalmanFilter()
#         self.landmarks=None
#         self.mp_hands=mp.solutions.hands
#         self.hands=self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             model_complexity=0,
#             min_detection_confidence=min_detection_conf,
#             min_tracking_confidence=min_tracking_conf
#         )
#         self.mouse_ctrl=MouseController()

#         self.img_buf=ImageSharedBuf((self.camera.height,self.camera.width,3))
#         self.shm_key=f'cv_image_{int(time.time())}'
#         self.shm=QSharedMemory(self.shm_key)
#         self.create_shm()
#         self.shm_send=False



#         self.text_img_need=False
#         self.photo_img_need=False
        
#     def create_shm(self)->None:
#         r_size=np.zeros((self.camera.height,self.camera.width,3),dtype=np.uint8).nbytes
#         if not self.shm.create(r_size):
#             if not self.shm.attach():
#                 print("create shared mem success")
    
#     def shm_need_send(self,is_need):
#         self.shm_send=is_need
    
#     def write_to_shm(self,image):
#         if self.img_buf.write_to_shm(self.shm,image):
#             self.shm_send_finish.emit()

  
#     def run(self):
#         cap=cv2.VideoCapture(self.camera.cam_id)
#         if not cap.isOpened():
#             print("camera open failed\r\n")
#             exit()
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.camera.width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.camera.height)
#         self.isRunning=True
#         while self.isRunning:
#             ret1,frame=cap.read()
#             if not ret1:
#                 print("camera read err")
#                 break
#             if self.shm_send:
#                 self.write_to_shm(frame)
#             rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#             landmarks=self.hands.process(rgb_frame)
#             filtered_landmarks=self.kf.run(landmarks)
#             is_click=self.mouse_ctrl.run(filtered_landmarks)
#             self.gesture_signal.emit(is_click,filtered_landmarks)  #发送信号
            
#         cap.release()

#     def send_img_to_text(self,need_send):
#         self.text_img_need=need_send
#         print(self.text_img_need)

#     def stop(self):
#         self.isRunning=False
#         self.hands.close()
#         self.wait()
#         if self.shm.isAttached():
#             self.shm.detach()
        








