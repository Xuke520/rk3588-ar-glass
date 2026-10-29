import math
import pyautogui
import numpy as np
import time
from PyQt6.QtCore import QThread

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

#默认输入数据均有效
class MouseController:
    def __init__(self,
                 screen_width=1920,
                 screen_height=1080,
                 smoothing=0.8):
        self.screen_width=screen_width
        self.screen_height=screen_height
        self.lx,self.ly=pyautogui.position()  #初始化为当前鼠标位置
        self.smoothing=smoothing
        self.click_count=0
        self.is_click=False

    def judge_click(self,landmarks,angle_range=(0.2*math.pi, 0.8*math.pi)):
        is_click=False
        angle_56_78=compute_angle(landmarks[5][0],landmarks[5][1],
                                landmarks[6][0],landmarks[6][1],
                                landmarks[7][0],landmarks[7][1],
                                landmarks[8][0],landmarks[8][1])
        if angle_range[0]<angle_56_78<angle_range[1]:
            dis_48=point_distance(landmarks[4][0],landmarks[4][1],
                                landmarks[8][0],landmarks[8][1])
            dis_26=point_distance(landmarks[2][0],landmarks[2][1],
                                landmarks[6][0],landmarks[6][1])
            dis_46=point_distance(landmarks[4][0],landmarks[4][1],
                                landmarks[6][0],landmarks[6][1])
            if dis_48<dis_26 and dis_48<dis_46 and dis_48<0.08:
                is_click=True
        return is_click
    
    def move(self,x,y):
        cx=int(x*self.screen_width)
        cy=int(y*self.screen_height)
        smooth_x=self.smoothing*cx+(1-self.smoothing)*self.lx
        smooth_y=self.smoothing*cy+(1-self.smoothing)*self.ly
        pyautogui.moveTo(smooth_x,smooth_y)
        self.lx,self.ly=smooth_x,smooth_y

    def click(self,landmarks):
        if self.judge_click(landmarks):
            self.click_count+=1
            if self.click_count>=2 and not self.is_click:
                print("   click    ")
                pyautogui.click()
                self.is_click=True
        else:
            self.click_count=0
            self.is_click=False

    def run(self,landmarks):
        x=landmarks[8][0]
        y=landmarks[8][1]
        self.move(x,y)
        self.click(landmarks)
        

class GestureMouseThread(QThread):
    def __init__(self,
                 screen_width=1920,
                 screen_height=1080):
        super().__init__()
        self.screen_width=screen_width
        self.screen_height=screen_height
        self.mouse=MouseController(screen_width=self.screen_width,
                                    screen_height=self.screen_height)
        self.isRunning=False
        self.landmarks=None
    
    "数据不需要加锁保护,这是因为该方法仅被QT的信号槽机制调用"
    "而Qt的信号槽机制已经保证了跨线程调用的安全性"
    def update_landmarks(self,landmarks):
        self.landmarks=landmarks
      
    def run(self):
        self.isRunning=True
        while self.isRunning:
            if self.landmarks:
                self.mouse.run(self.landmarks)
            time.sleep(0.03)

    def stop(self):
        self.isRunning=False
        self.wait()

