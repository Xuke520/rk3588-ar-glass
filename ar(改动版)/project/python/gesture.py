import cv2
import time
import struct
import threading
import numpy as np
import mediapipe as mp
from multiprocessing import shared_memory





class Camera():
    def __init__(self,cam_id:int=0,width:int=640,height:int=480):
        self.cam_id=cam_id
        self.width=width
        self.height=height

class GestureThread(threading.Thread):
    def __init__(self,camera:Camera,
                 max_hands:int=1,
                 min_detection_conf:float=0.75,
                 min_tracking_conf:float=0.75):
        threading.Thread.__init__(self)
        self.camera=camera
        self.max_hands=max_hands
        self.results=None
        self.frame=None
        self.running=False
        self.lock=threading.Lock()  # 保护共享数据
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

    def run(self):
        try:
            cap=cv2.VideoCapture(self.camera.cam_id)
            if not cap.isOpened():
                print("camera open failed\r\n")
                exit()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.camera.height)
            self.running=True
            while self.running:
                ret,frame=cap.read()
                if not ret:
                    break       
                rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=self.hands.process(rgb_frame)
                with self.lock:
                    self.results=results
        finally:
            cap.release()
            self.hands.close()

    def stop(self):
        self.running=False
        self.join()

    def get_landmarks(self):
        with self.lock:
            return self.results
    

class SharedMemory():
    def __init__(self,name,size,mode=0o666):
        self.name=f"{name}"  
        self.size=size      #手势数据尺寸max_hands*21*3*4
        self.mode=mode
        self.shared_mem=None

    def create(self):
        try:
            self.shared_mem=shared_memory.SharedMemory(
                            name=self.name, 
                            create=True, 
                            size=self.size+4
                        )
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to create shared memory:{e}")

    def write(self,data):
        if not isinstance(data,bytes):
            raise TypeError("data must be bytes")
        if len(data)>self.size:
            raise ValueError("Data exceeds shared memory size")
        self.shared_mem.buf[:len(data)]=data   
        return True
    
    #将手势数据写入共享内存,数据格式:数据总长度(4字节)+具体手势数据
    def write_landmarks(self,landmarks):
        if not landmarks or not landmarks.multi_hand_landmarks:
            self.shared_mem.buf[:(self.size+4)]= b'\x00'*(self.size+4)
        else:
            data_content=bytearray()#存储手势数据
            data=bytearray()#创建最终数据:先写入长度,再写入手势数据
            data.extend(struct.pack('I',self.size))  #在开头写入长度
            for hand in landmarks.multi_hand_landmarks: 
                for idx,landmark in enumerate(hand.landmark):  
                    #每个坐标用4字节float表示 (x,y,z)
                    data_content.extend(struct.pack('fff',landmark.x,landmark.y,landmark.z))
            data.extend(data_content)  
            self.shared_mem.buf[:(self.size+4)]=data   
        time.sleep(0.00001)#延时,否则内存不更新

    def read(self,r_size):
        if not self.shared_mem:
            raise RuntimeError("Shared memory not initialized")
        return bytes(self.shared_mem.buf)
    
    def destroy(self):
        self._cleanup()
    
    def _cleanup(self):
        if self.shared_mem:
            self.shared_mem.close()
            try:
                self.shared_mem.unlink()
            except FileNotFoundError:
                pass

