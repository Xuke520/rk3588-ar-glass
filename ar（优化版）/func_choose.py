from neural_net.rknn.ppocr.ppocr import PPOCRRKNN
from neural_net.cloud.gummy.gummy import GummyCloud
import cv2
import time
import numpy as np
from hand_gesture import Camera,ImageSharedBuf
from network_tools import RTSPReceiver,SyncUDPSender
from PyQt6.QtCore import pyqtSignal,QThread,QMutex,QMutexLocker,QSharedMemory


class FuncThread(QThread):
    shm_need_send=pyqtSignal(bool)
    shm_read_ready=pyqtSignal()
    def __init__(self,
                 camera:Camera):
        super().__init__()

        self.camera=camera
        self.gummy=GummyCloud()
        self.rtsp=RTSPReceiver()
        self.remote=SyncUDPSender()
        self.ppocr=PPOCRRKNN()
        # self.yolov5=Yolov5RKNN()
        self.func_id=0   
        self.running=False
        self.mutex=QMutex()
        self.shared_image_state=False
        self.image=None
        self.shm_key=f'cv_image_{int(time.time())}'
        # self.shm_key=f'cv_image'
        self.shm=QSharedMemory(self.shm_key)
        self.img_buf=None

    
    def attach_shm(self):
        for i in range(10):
            if self.shm.attach():
                self.img_buf=ImageSharedBuf((self.camera.height,self.camera.width,3))
                print(f"共享内存附加成功")
                return True
            print(f"共享内存附加失败")
            time.sleep(0.1)
        return False
  

    def choose_func(self,func_id:int)->None:
        with QMutexLocker(self.mutex):
            self.func_id=func_id
    
    def read_from_shm(self): 
        if self.img_buf.read_from_shm(self.shm):  
            with QMutexLocker(self.mutex):
                self.image=self.img_buf.array
    
    def run(self):
        if not self.attach_shm():
            self.shm_need_send.emit(False)
            raise RuntimeError("无法附加到共享内存，请确保写入线程已启动")
        func_id=0
        self.running=True
        while self.running:
            with QMutexLocker(self.mutex):
                func_id=self.func_id
            if func_id<4:
                self.shm_need_send.emit(False)
            else:
                self.shm_need_send.emit(True)

            
            if(func_id==0):
                pass
            elif(func_id==1):#gummy
                text=self.gummy.infer()
                print(text)
            elif(func_id==2):
                image=self.rtsp.get_frame()
                if image is not None:
                    cv2.imshow("rtsp",image)
            elif(func_id==3):
                pass
     
            elif(func_id==4):
                with QMutexLocker(self.mutex):
                    image=self.image
                if image is not None:
                    pass
                filter_boxes,filter_rec_res=self.ppocr.infer(image)
                cv2.imshow("1",image)

            elif(func_id==5):
                print("success")
                with QMutexLocker(self.mutex):
                    image=self.image
                if image is not None:
                    print("no image")
                    pass
                cv2.imshow("1",image)

            cv2.waitKey(3)
    


    def stop(self):
        with QMutexLocker(self.mutex):
            self.func_id=0
        self.running=False
  
        self.ppocr.release()
        self.gummy.release()
        self.rtsp.release()
        self.wait()
        if self.shm.isAttached():
            self.shm.detach()


        
    

if __name__=='__main__':
    nt=FuncThread()
    nt.start()
