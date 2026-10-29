import os
import sys
import cv2
import copy
import numpy as np

realpath=os.path.abspath(__file__)
_sep=os.path.sep
realpath=realpath.split(_sep)
sys.path.append(
    os.path.join(
        realpath[0]+_sep,  
        *realpath[1:realpath.index('rknn')+1]  
    )
)

import ppocr.utils.operators as operators
from ppocr.utils.rec_postprocess import CTCLabelDecode
from ppocr.utils.db_postprocess import DBPostProcess,DetPostProcess
from py_utils.rknn_executor import RKNN_model_container 
from py_utils.rknn_executor import RKNN_model_container 


CHARACTER_DICT_PATH=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocr_keys_v1.txt"

REC_PRE_PROCESS_CONFIG=[ 
    {
        'NormalizeImage': {
            'std': [1,1,1],
            'mean': [0,0,0],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
]

DET_PRE_PROCESS_CONFIG=[
    {
        'DetResizeForTest':{
                'image_shape':[480,480]
            }
        },
    {
        'NormalizeImage': 
        {
                'std': [1.,1.,1.],
                'mean': [0.,0.,0.],
                'scale': '1.',
                'order': 'hwc'
        }
    }
]


POSTPROCESS_CONFIG=[
    {#rec
        'CTCLabelDecode':{
            "character_dict_path": CHARACTER_DICT_PATH,
            "use_space_char": True
            }   
    },
    {#det
    'DBPostProcess':{
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
        }
    }
]




def get_rotate_crop_image(img,points):
    assert len(points) ==4,"shape of points must be 4*2"
    img_crop_width=int(
        max(
            np.linalg.norm(points[0]-points[1]),
            np.linalg.norm(points[2]-points[3])))
    img_crop_height=int(
        max(
            np.linalg.norm(points[0]-points[3]),
            np.linalg.norm(points[1]-points[2])))
    pts_std=np.float32([[0,0],[img_crop_width,0],
                          [img_crop_width,img_crop_height],
                          [0,img_crop_height]])
    M=cv2.getPerspectiveTransform(points,pts_std)
    dst_img=cv2.warpPerspective(
        img,
        M,(img_crop_width,img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height,dst_img_width=dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >=1.5:
        dst_img=np.rot90(dst_img)
    return dst_img

def sorted_boxes(dt_boxes):
    num_boxes=dt_boxes.shape[0]
    sorted_boxes=sorted(dt_boxes,key=lambda x: (x[0][1],x[0][0]))
    _boxes=list(sorted_boxes)

    for i in range(num_boxes-1):
        for j in range(i,-1,-1):
            if abs(_boxes[j+1][0][1]-_boxes[j][0][1]) < 10 and \
                    (_boxes[j+1][0][0] < _boxes[j][0][0]):
                tmp=_boxes[j]
                _boxes[j]=_boxes[j+1]
                _boxes[j+1]=tmp
            else:
                break
    return _boxes


class TextDetector:
    def __init__(self,
                 det_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_det_rk3566_int8.rknn",
                 target:str='rk3566'):
        self.model=RKNN_model_container(det_model_path,target,None)
        self.preprocess_funct=[]
        for item in DET_PRE_PROCESS_CONFIG:
            for key in item:
                pclass=getattr(operators,key)
                p=pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess=DBPostProcess(**POSTPROCESS_CONFIG[1]['DBPostProcess'])
        self.det_postprocess=DetPostProcess()

    def preprocess(self,img):
        for p in self.preprocess_funct:
            img=p(img)
        image_input=img['image']
        image_input=np.expand_dims(image_input,0)
        img['image']=image_input
        return img

    def infer(self,img):
        model_input=self.preprocess({'image':img})
        output=self.model.infer([model_input['image']])
        preds={'maps':output[0].astype(np.float32)}
        result=self.db_postprocess(preds,model_input['shape'])
        output=self.det_postprocess.filter_tag_det_res(result[0]['points'],img.shape)
        return output
    
    def release(self)->None:
        if self.model:
            self.model.release()
            self.model=None
    

    
    
class TextRecognizer:
    def __init__(self,
                 rec_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_rec_rk3566_fp16.rknn",
                 target:str='rk3566'):
        self.model=RKNN_model_container(rec_model_path,target,None)
        self.preprocess_funct=[]
        for item in REC_PRE_PROCESS_CONFIG:
            for key in item:
                pclass=getattr(operators,key)
                p=pclass(**item[key])
                self.preprocess_funct.append(p)
        #初始化后处理
        self.ctc_postprocess=CTCLabelDecode(**POSTPROCESS_CONFIG[0]['CTCLabelDecode'])

    def preprocess(self,img_dict):
        img=img_dict['image']
        for p in self.preprocess_funct:
            img=p({'image': img})['image']
        img=np.expand_dims(img,axis=0)
        return {'image':img}

    #处理单张图片
    # def run(self,img):
    #     model_input=self.preprocess({'image':img})
    #     output=self.model.infer([model_input['image']])
    #     preds=output[0].astype(np.float32)
    #     output=self.ctc_postprocess(preds)
    #     return output
    
    #处理多张图片
    def infer(self,imgs):
        outputs=[]
        for img in imgs:
            img=cv2.resize(img,(320,48))
            model_input=self.preprocess({'image':img})
            output=self.model.infer([model_input['image']])
            # preds={'maps':output[0].astype(np.float32)}
            preds=output[0].astype(np.float32)
            output=self.ctc_postprocess(preds)
            outputs.append(output)
        return outputs
    
    def release(self)->None:
        if self.model:
            self.model.release()
            self.model=None




class PPOCRRKNN:
    def __init__(self,
                 det_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_det_rk3566_fp16.rknn",
                 rec_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_rec_rk3566_fp16.rknn",
                 target:str='rk3566'):
        self.text_detector=TextDetector(det_model_path,target)
        self.text_recognizer=TextRecognizer(rec_model_path,target)
        self.drop_score=0.5
    
    def infer(self,img):
        ori_img=img.copy()
        dt_boxes=self.text_detector.infer(img)
        if dt_boxes is None:
            return None,None
        img_crop_list=[]
        dt_boxes=sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box=copy.deepcopy(dt_boxes[bno])
            img_crop=get_rotate_crop_image(ori_img,tmp_box)
            img_crop_list.append(img_crop)
        rec_res=self.text_recognizer.infer(img_crop_list)
        filter_boxes,filter_rec_res=[],[]
        for box,rec_result in zip(dt_boxes,rec_res):
            text,score=rec_result[0]
            if score >=self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes,filter_rec_res
    
    def release(self)->None:
        if self.text_detector:
            self.text_detector.release()
        if self.text_recognizer:
            self.text_recognizer.release()

    

if __name__ =='__main__':
    sys=PPOCRRKNN()
    cap=cv2.VideoCapture(28)
    if not cap.isOpened():
        print("camera open failed\r\n")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while True:
        ret,frame=cap.read()
        if not ret:
            print("camera read err")
            break
        filter_boxes,filter_rec_res=sys.infer(frame)
        print(filter_rec_res)
        # cv2.imshow('Object Detection',result_frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break


