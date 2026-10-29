import os
import sys
import cv2
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

realpath=os.path.abspath(__file__)
_sep=os.path.sep
realpath=realpath.split(_sep)
sys.path.append(
    os.path.join(
        realpath[0]+_sep,  
        *realpath[1:realpath.index('rknn')+1]  
    )
)

import utils.operators as operators
from utils.rec_postprocess import CTCLabelDecode
from utils.db_postprocess import DBPostProcess,DetPostProcess
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
                 det_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_det_rk3588_int8.rknn",
                 target:str='rk3588'):
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
                 rec_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_rec_rk3588_fp16.rknn",
                 target:str='rk3588'):
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
                 det_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_det_rk3588_fp16.rknn",
                 rec_model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/ppocrv4_rec_rk3588_fp16.rknn",
                 target:str='rk3588'):
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
            print(text)
            if score >=self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes,filter_rec_res
    
    def release(self)->None:
        if self.text_detector:
            self.text_detector.release()
        if self.text_recognizer:
            self.text_recognizer.release()


# def draw_ocr_results(image, boxes, texts):
#     """兼容RK3588的中文显示解决方案"""
#     # 转换为PIL格式
#     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(image_pil)
    
#     # 多路径尝试加载字体
#     font = None
#     font_paths = [
#         "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿
#         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto字体
#         os.path.join(os.path.dirname(__file__), "fonts/simhei.ttf")  # 项目自带
#     ]
    
#     for path in font_paths:
#         try:
#             font = ImageFont.truetype(path, 20)
#             break
#         except:
#             continue
    
#     if font is None:
#         font = ImageFont.load_default()
    
#     # 统一使用PIL绘制
#     for box, text in zip(boxes, texts):
#         box = [tuple(p.astype(int)) for p in box]
        
#         # 绘制边界框
#         draw.polygon(box, outline=(0, 255, 0), width=2)
        
#         # 处理文本
#         text_content = str(text[0][0]).encode('utf-8').decode('utf-8', 'ignore')
#         text_x = min(box[0][0], box[2][0])
#         text_y = min(box[0][1], box[1][1]) - 25
        
#         # 绘制背景和文本
#         text_bbox = draw.textbbox((text_x, text_y), text_content, font=font)
#         draw.rectangle(text_bbox, fill=(0, 0, 255))
#         draw.text((text_x, text_y), text_content, font=font, fill=(255, 255, 255))
    
#     return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def draw_ocr_results(image, boxes, texts):
    """将识别文字绘制在文本框内部的解决方案"""
    # 转换为PIL格式
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # 字体加载（保持原有逻辑）
    font = None
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        os.path.join(os.path.dirname(__file__), "fonts/simhei.ttf")
    ]
    
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 20)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # 统一使用PIL绘制
    for box, text in zip(boxes, texts):
        box = [tuple(p.astype(int)) for p in box]
        
        # 绘制边界框（绿色）
        draw.polygon(box, outline=(0, 255, 0), width=2)
        
        # 处理文本内容
        text_content = str(text[0][0]).encode('utf-8').decode('utf-8', 'ignore')
        
        # 计算文本框的边界
        min_x = min(p[0] for p in box)
        max_x = max(p[0] for p in box)
        min_y = min(p[1] for p in box)
        max_y = max(p[1] for p in box)
        
        # 计算文本位置（居中显示）
        left, top, right, bottom = font.getbbox(text_content)
        text_width = right - left
        text_height = bottom - top
        text_x = min_x + (max_x - min_x - text_width) // 2
        text_y = min_y + (max_y - min_y - text_height) // 2
        
        # 确保文本不会超出框边界
        text_x = max(min_x, min(text_x, max_x - text_width))
        text_y = max(min_y, min(text_y, max_y - text_height))
        
        # 绘制半透明背景（增强可读性）
        bg_box = (
            text_x - 2, text_y - 2,
            text_x + text_width + 2,
            text_y + text_height + 2
        )
        bg_layer = Image.new('RGBA', image_pil.size, (0,0,0,0))
        bg_draw = ImageDraw.Draw(bg_layer)
        bg_draw.rectangle(bg_box, fill=(0, 0, 255, 128))  # 半透明蓝色背景
        image_pil = Image.alpha_composite(image_pil.convert('RGBA'), bg_layer)
        draw = ImageDraw.Draw(image_pil)
        
        # 绘制文本（白色）
        draw.text((text_x, text_y), text_content, font=font, fill=(255, 255, 255))
    
    return cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

if __name__ =='__main__':
    ocr_sys = PPOCRRKNN()
    img_path = "test.jpg"  # 确保有一张测试图片
    if not os.path.exists(img_path):
        print(f"请放置测试图片: {img_path}")
        exit()
    
    frame = cv2.imread(img_path)
    if frame is None:
        print("图片读取失败")
        exit()
    
    filter_boxes, filter_rec_res = ocr_sys.infer(frame)
    
    if filter_boxes is not None and filter_rec_res is not None:
        print(f"\n识别到 {len(filter_boxes)} 个文字区域:")
        for i, (box, rec) in enumerate(zip(filter_boxes, filter_rec_res)):
            text, score = rec[0]
            print(f"  [{i}] {text} (置信度: {score:.3f})")
        
        # 绘制结果
        result_frame = draw_ocr_results(frame.copy(), filter_boxes, filter_rec_res)
        cv2.imwrite('ocr_result.jpg', result_frame)
        print("\n结果已保存到 ocr_result.jpg")
        
        # 显示结果
        cv2.imshow('OCR Result', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到文字区域")
    
    ocr_sys.release()


