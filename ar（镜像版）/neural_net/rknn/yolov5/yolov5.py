import os
import sys
import cv2
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


from py_utils.rknn_executor import RKNN_model_container 
from py_utils.coco_utils import COCO_test_helper




OBJ_THRESH=0.25
NMS_THRESH=0.45
IMG_SIZE=(640,640)  # (width,height),such as (1280,736)
CLASSES=("person","bicycle","car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier","toothbrush ")

coco_id_list=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,
         35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
         64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]


def filter_boxes(boxes,box_confidences,box_class_probs):
    box_confidences=box_confidences.reshape(-1)
    class_max_score=np.max(box_class_probs,axis=-1)
    classes=np.argmax(box_class_probs,axis=-1)
    _class_pos=np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores=(class_max_score* box_confidences)[_class_pos]
    boxes=boxes[_class_pos]
    classes=classes[_class_pos]
    return boxes,classes,scores

def nms_boxes(boxes,scores):
    x=boxes[:,0]
    y=boxes[:,1]
    w=boxes[:,2] - boxes[:,0]
    h=boxes[:,3] - boxes[:,1]
    areas=w * h
    order=scores.argsort()[::-1]
    keep=[]
    while order.size > 0:
        i=order[0]
        keep.append(i)
        xx1=np.maximum(x[i],x[order[1:]])
        yy1=np.maximum(y[i],y[order[1:]])
        xx2=np.minimum(x[i] + w[i],x[order[1:]] + w[order[1:]])
        yy2=np.minimum(y[i] + h[i],y[order[1:]] + h[order[1:]])
        w1=np.maximum(0.0,xx2 - xx1 + 0.00001)
        h1=np.maximum(0.0,yy2 - yy1 + 0.00001)
        inter=w1 * h1
        ovr=inter / (areas[i] + areas[order[1:]] - inter)
        inds=np.where(ovr <= NMS_THRESH)[0]
        order=order[inds + 1]
    keep=np.array(keep)
    return keep

def box_process(position,anchors):
    grid_h,grid_w=position.shape[2:4]
    col,row=np.meshgrid(np.arange(0,grid_w),np.arange(0,grid_h))
    col=col.reshape(1,1,grid_h,grid_w)
    row=row.reshape(1,1,grid_h,grid_w)
    grid=np.concatenate((col,row),axis=1)
    stride=np.array([IMG_SIZE[1]//grid_h,IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    col=col.repeat(len(anchors),axis=0)
    row=row.repeat(len(anchors),axis=0)
    anchors=np.array(anchors)
    anchors=anchors.reshape(*anchors.shape,1,1)
    box_xy=position[:,:2,:,:]*2 - 0.5
    box_wh=pow(position[:,2:4,:,:]*2,2) * anchors
    box_xy += grid
    box_xy *= stride
    box=np.concatenate((box_xy,box_wh),axis=1)
    xyxy=np.copy(box)
    xyxy[:,0,:,:]=box[:,0,:,:] - box[:,2,:,:]/ 2  # top left x
    xyxy[:,1,:,:]=box[:,1,:,:] - box[:,3,:,:]/ 2  # top left y
    xyxy[:,2,:,:]=box[:,0,:,:] + box[:,2,:,:]/ 2  # bottom right x
    xyxy[:,3,:,:]=box[:,1,:,:] + box[:,3,:,:]/ 2  # bottom right y
    return xyxy

def post_process(input_data,anchors):
    boxes,scores,classes_conf=[],[],[]
    # 1*255*h*w -> 3*85*h*w
    input_data=[_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:],anchors[i]))
        scores.append(input_data[i][:,4:5,:,:])
        classes_conf.append(input_data[i][:,5:,:,:])

    def sp_flatten(_in):
        ch=_in.shape[1]
        _in=_in.transpose(0,2,3,1)
        return _in.reshape(-1,ch)

    boxes=[sp_flatten(_v) for _v in boxes]
    classes_conf=[sp_flatten(_v) for _v in classes_conf]
    scores=[sp_flatten(_v) for _v in scores]

    boxes=np.concatenate(boxes)
    classes_conf=np.concatenate(classes_conf)
    scores=np.concatenate(scores)

    # filter according to threshold
    boxes,classes,scores=filter_boxes(boxes,scores,classes_conf)

    # nms
    nboxes,nclasses,nscores=[],[],[]

    for c in set(classes):
        inds=np.where(classes == c)
        b=boxes[inds]
        c=classes[inds]
        s=scores[inds]
        keep=nms_boxes(b,s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None,None,None

    boxes=np.concatenate(nboxes)
    classes=np.concatenate(nclasses)
    scores=np.concatenate(nscores)

    return boxes,classes,scores

def draw(image,boxes,scores,classes):
    for box,score,cl in zip(boxes,scores,classes):
        top,left,right,bottom=[int(_b) for _b in box]
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl],top,left,right,bottom,score))
        cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)
        cv2.putText(image,'{0} {1:.2f}'.format(CLASSES[cl],score),
                    (top,left - 6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)


class Yolov5RKNN:
    def __init__(self,
                 model_path:str=f"{os.path.dirname(os.path.abspath(__file__))}/model/yolov5_rk3588_fp16.rknn",
                 target:str='rk3588'):
        self.model=RKNN_model_container(model_path,target,None)
        self.co_helper=COCO_test_helper(enable_letter_box=True)
        self.anchors=self.load_anchors(f"{os.path.dirname(os.path.abspath(__file__))}/model/anchors_yolov5.txt")

    def load_anchors(self,anchors_path):
        with open(anchors_path,'r') as f:
            values=[float(_v) for _v in f.readlines()]
            return np.array(values).reshape(3,-1,2).tolist()
        
    def preprocess(self,image):
        img=self.co_helper.letter_box(im=image.copy(),
                                      new_shape=(IMG_SIZE[1],IMG_SIZE[0]),
                                      pad_color=(0,0,0))
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # def infer(self,image):
    #     img=self.preprocess(image)
    #     outputs=self.model.infer([np.expand_dims(img,0)])
    #     #outputs=self.model.run([img])
    #     boxes,classes,scores=post_process(outputs,self.anchors)
    #     if boxes is not None:
    #         draw(image,self.co_helper.get_real_box(boxes),scores,classes)
    #     return image,classes,scores

    def infer(self,image):
        img=self.preprocess(image)
        outputs=self.model.infer([np.expand_dims(img,0)])
        #outputs=self.model.run([img])
        boxes,classes,scores=post_process(outputs,self.anchors)
        if boxes is not None:
            target_id=0
            person_indices = [i for i, cls in enumerate(classes) if cls == target_id]
            person_boxes = boxes[person_indices]
            person_scores = scores[person_indices]
            person_classes = classes[person_indices]

            # 只绘制人形检测框
            if len(person_boxes) > 0:
                draw(image, self.co_helper.get_real_box(person_boxes), 
                     person_scores, person_classes)

        return image



    def release(self)->None:
        if self.model:
            self.model.release()
            self.model=None




# if __name__ == '__main__':
#     yolo=Yolov5RKNN()
#     cap=cv2.VideoCapture(21)
#     if not cap.isOpened():
#         print("camera open failed\r\n")
#         exit()
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#     while True:
#         ret,frame=cap.read()
#         if not ret:
#             print("camera read err")
#             break
#         result_frame= yolo.infer(frame)
#         cv2.imshow('Object Detection',result_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break



if __name__ == '__main__':
    yolo = Yolov5RKNN()
    cap = cv2.VideoCapture(21)
    
    if not cap.isOpened():
        print("摄像头打开失败")
        exit()
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 获取摄像头帧率和分辨率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # 默认帧率
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出目录
    output_dir = "./output_videos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"detection.mp4")
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取错误")
                break
            
            # 执行检测并获取结果图像
            result_frame = yolo.infer(frame)
            
            # 显示实时结果
            cv2.imshow('Object Detection', result_frame)
            
            # 写入视频文件
            out.write(result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"视频已保存到: {output_path}")