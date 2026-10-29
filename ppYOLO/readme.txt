基于PPYolo的二维码检测和识别_副本
公开
基于PPYolo实现复杂场景下的二维码高精度检测和识别，二维码尺度、角度多样，其中检测模块由PPYolo实现，识别模块由zbar库实现。本项目包含完整的训练和测试方法
2026-07-06 15:14:18
计算机视觉
AI Studio 经典版
JupyterLab
2.1.2
Python3
版本内容
数据集
后台任务
服务部署
当前版本：
草稿
07-06 15:15:15
当前内容阅读耗时约38分钟，试试
小桨总结
一、任务概述
近几年，随着移动支付的普及，二维码技术得到了广泛的使用。二维码的信息容纳性高、使用便利，可以使用非接触式方式快速实现信息认证交互，因此二维码成为了目前最为普遍的编码方式。二维码使用范围很广，小至日常购物支付、出入搭乘交通，大至工厂的物流、订单管理都可以见到二维码的身影（例如给流水线产品打二维码标签实现物品快速认证和唯一性管理）。由于真实环境中场景多样和复杂，对二维码的检测与识别技术带来了较高的要求，包括识别的精确度和时效性。 然而，目前采用的二维码检测与识别技术仍有欠缺，对于规范化场景下的二维码用传统的检测技术可以达到较高的精度，但是一旦场景、拍摄角度、光线发生变化，传统方法就显得力不从心。本项目从深度学习方法切入，实现基于深度学习的二维码检测与识别，利用深度学习算法提升二维码检测精度。

本项目使用百度飞桨深度学习框架来实现，飞桨版本为PaddlePaddle2.1动态图版本, Python版本是3.7.4。本文将全流程详细讲解训练和测试步骤，检测部分采用飞桨PaddleDetection套件的PPYolo目标检测算法实现，解码识别部分采用Zbar库实现。有些人可能会有疑惑，为什么不直接用Zbar库进行检测和解码呢？这是因为由于实际工业场景中二维码目标往往较小、模糊或倾斜，因此直接用原始拍摄的图片输送给Zbar很难得到解码结果，因此我们需要预先将二维码检测出来然后抠取二维码区域再送给Zbar，从而提升检测和识读有效性。

数据集由两部分组成，一部分是我从网上收集并筛选的，另一部分是向文献【1】的作者申请得到的，两个数据集均采用Labelme的矩形框进行标注，标注精度较高。

本项目也作为飞桨领航团AI达人创造营的个人作业的一部分，有兴趣的读者可以关注并folk本项目。



二、算法介绍
目标检测技术作为计算机视觉的主要方向之一，应用非常广泛。除了可以直接进行目标检测，同时也是人脸检测，行人检测，文字检测、关键点检测、实例分割等复杂应用的基础技术。而说到目标检测算法，就不得不提 YOLO 系列。YOLO 作为单阶段模型的代表之一，相较于更早提出的两阶段目标检测算法，不仅拥有更快的预测速度；对于背景图像（非物体）中的部分被包含在候选框的情况误检率更低，还拥有更好的算法通用性。

PP-YOLO 是飞桨团队在 YOLOv3 的基础上又做了一系列深度优化的成果。通过添加尽量不引入额外计算量的 trick 来提升 YOLOv3 模型的精度，最终在 COCO 数据集上达到了 45.9% 的精度；同时在 V100 单卡上达到了 72.9 FPS；在开启 TensorRT 下，FP16 的推理速度达到 155.6FPS。这使得 PP-YOLO 在精度和速度上达到了最佳平衡，全面超越原生 YOLOv4 模型，成为产业界最佳的目标检测模型。下图显示了具体的算法性能对比：



为了兼顾 YOLOv3 模型的精度和预测速度，打造精度速度高性价比的工业实用模型，PP-YOLO 使用了精度速度更优的带可变形卷积的 ResNet50vd 模型（ResNet50vd-DCN），预测速度提升 20%，精度也有小量提升。在此基础上，加入 Coord Conv 和 SPP 等运算量很小，但能有效提升特征提取效率的 trick。另外，还引入了基于 IoU 优化检测框定位精度的 IoU Loss，IoU Aware 等方法。这些方法基本不引入额外的计算，但能有效地提高输出预测框的定位精度，有效解决 YOLOv3 模型定位精度不高的问题。在预测框后处理上，引入 Grid Sensitive 使得预测框学习和解码更加稳定，同时 Matrix NMS 能更加快速有效地完成预测框的非极大值抑制，更高效的保留高质量的预测框，滤除低质量的预测框。在训练过程中，PP-YOLO 模型使用 DropBlock 这种更适用于目标检测的 dropout 方法来减小过拟合，同时使用 EMA（指数滑动平均）的训练方法，即使用梯度历史加权平均值来平滑掉难样本或者误差样本带来的梯度抖动，让训练过程更加平滑稳定。

通过上述策略对 YOLOv3 模型进行深度优化，PP-YOLO 模型在精度和速度上都大幅优于 YOLOv3 模型，甚至优于更新的 YOLOv4 模型。因此 PP-YOLO 可以说是一个非常适用于工业项目的高性价比实用性模型。

三、训练和评估
3.1 环境配置
本项目由于在AI Studio上训练和运行，因此默认已经安装了PaddlePaddle。如果读者在本地进行训练，那么需要先安装PaddlePaddle，本项目使用的是PaddlePaddle2.1版本。

3.1.1 下载并安装PaddleDetection
In [ ]
# 下载PaddleDetection。由于github上资源比较慢，这里建议使用gitee上的资源
!git clone https://gitee.com/paddlepaddle/PaddleDetection.git -b release/2.1
In [2]
# 安装PaddleDetection（该命令有可能因为网络原因第1次安装不成功，需要执行多次直至成功）
%cd /home/aistudio/PaddleDetection
!python setup.py install
3.1.2 验证是否安装成功
In [ ]
%cd /home/aistudio/PaddleDetection
!python ppdet/modeling/tests/test_architectures.py
输出如下所示内容说明验证成功：

Ran 5 tests in 3.022s
OK
3.2 数据集处理
数据集图片总数1994张，涵盖多种背景、角度、光线环境、距离，部分样例如下图所示：



该数据集我们采用labelme工具并且使用矩形框进行标注，适合目标检测算法进行训练。该数据集解压后目录结构如下：



其中，imgs文件夹存放jpg格式图片，labels文件夹存放对应的json格式标注文件，其余的几个.py文件是用于前期处理数据集的，读者可以将其忽略。

下面将数据集提取到PaddleDetection/dataset目录下并对其进行解压：

In [ ]
%cd /home/aistudio
# 拷贝数据集到PaddleDetection/dataset目录下
!cp data/data103078/qr.zip PaddleDetection/dataset/    
# 解压
%cd /home/aistudio/PaddleDetection/dataset
!unzip qr.zip
由于本项目使用PaddleDetection的PPYolo算法进行训练，为了方便数据处理，我们将我们标注的labelme格式转成coco数据集格式。

lableme格式数据集转coco格式

这里使用PaddleDetection现成的工具函数进行格式转换，其中我们将整个数据集按照训练集、验证集、测试集进行划分，划分比例为0.8、0.1、0.1：

In [1]
%cd /home/aistudio/PaddleDetection
!python tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./dataset/qr/labels/ \
                --image_input_dir ./dataset/qr/imgs/ \
                --output_dir ./dataset/qr_coco/ \
                --train_proportion 0.8 \
                --val_proportion 0.1 \
                --test_proportion 0.1
转换后在PaddleDetection/dataset目录下会生成一个名为qr_coco的文件夹，该文件夹就是转换过后的coco数据集格式的二维码数据集：



其中，train、val、test文件夹分别用于存放训练集、验证集、测试集图片，annotations用来存放标注文件。简单来看，coco数据集格式就是将所有标注的单一json文件合并为1个完整的json文件。

3.3 训练
新建一个ppyolo_mbv3_large_qr.yml文件到PaddleDetection/configs目录下面，内容如下：

use_gpu: true
epoch: 100
log_iter: 20
save_dir: output
snapshot_epoch: 10
weights: output/ppyolo_mbv3_large_qr/79
metric: COCO
num_classes: 1

TrainDataset:
  !COCODataSet
    image_dir: train
    anno_path: annotations/instance_train.json
    dataset_dir: dataset/qr_coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val
    anno_path: annotations/instance_val.json
    dataset_dir: dataset/qr_coco

TestDataset:
  !ImageFolder
    image_dir: test
    anno_path: annotations/instance_test.json
    dataset_dir: dataset/qr_coco


architecture: YOLOv3
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x1_0_ssld_pretrained.pdparams
norm_type: sync_bn
use_ema: true
ema_decay: 0.9998

YOLOv3:
  backbone: MobileNetV3
  neck: PPYOLOFPN
  yolo_head: YOLOv3Head
  post_process: BBoxPostProcess

MobileNetV3:
  model_name: large
  scale: 1.
  with_extra_blocks: false
  extra_block_filters: []
  feature_maps: [13, 16]

PPYOLOFPN:
  in_channels: [160, 368]
  coord_conv: true
  conv_block_num: 0
  spp: true
  drop_block: true

YOLOv3Head:
  anchors: [[109, 145],[233, 103],[175, 247],[336, 194],[228, 362],[376, 395]]
  anchor_masks: [[3, 4, 5], [0, 1, 2]]
  loss: YOLOv3Loss

YOLOv3Loss:
  ignore_thresh: 0.5
  downsample: [32, 16]
  label_smooth: false
  scale_x_y: 1.05
  iou_loss: IouLoss

IouLoss:
  loss_weight: 2.5
  loss_square: true

BBoxPostProcess:
  decode:
    name: YOLOBox
    conf_thresh: 0.005
    downsample_ratio: 32
    clip_bbox: true
    scale_x_y: 1.05
  nms:
    name: MultiClassNMS
    keep_top_k: 100
    nms_threshold: 0.45
    nms_top_k: 1000
    score_threshold: 0.005

worker_num: 2
TrainReader:
  inputs_def:
    num_max_boxes: 90
  sample_transforms:
    - Decode: {}
    - Mixup: {alpha: 1.5, beta: 1.5}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize:
        target_size: [[288,288],[384,384],[512,512],[640,640],[768,768]]
        random_size: True
        random_interp: True
        keep_ratio: False
    - NormalizeBox: {}
    - PadBox: {num_max_boxes: 90}
    - BboxXYXY2XYWH: {}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
    - Gt2YoloTarget:
        anchor_masks: [[3, 4, 5], [0, 1, 2]]
        anchors: [[109, 145],[233, 103],[175, 247],[336, 194],[228, 362],[376, 395]]
        downsample_ratios: [32, 16]
        iou_thresh: 0.25
        num_classes: 1
  batch_size: 28
  mixup_epoch: 200
  shuffle: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [512, 512], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  batch_size: 28

TestReader:
  inputs_def:
    image_shape: [3, 512, 512]
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [512, 512], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  batch_size: 1

LearningRate:
  base_lr: 0.0025
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones:
    - 430
    - 540
    - 610
  - !LinearWarmup
    start_factor: 0.
    steps: 4000

OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0005
    type: L2
实际训练前需要根据显卡显存情况调整上述batchsize参数，本项目使用V100 32G显存的显卡进行训练。

In [ ]
# 开始训练
%cd /home/aistudio/PaddleDetection
!export CUDA_VISIBLE_DEVICES=0
!python tools/train.py -c configs/ppyolo_mbv3_large_qr.yml
本项目训练到大概85个epoch时手动停止了，因为loss损失感觉没有太大变化了，整个训练时间大概花了9个小时左右。

3.4 评估
In [ ]
%cd /home/aistudio/PaddleDetection
!export CUDA_VISIBLE_DEVICES=0 
!python tools/eval.py -c configs/ppyolo_mbv3_large_qr.yml -o weights=./output/ppyolo_mbv3_large_qr/79.pdparams
评估结果如下

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.813
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.964
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.653
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.820
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.824
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.861
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.861
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.859
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.869
[08/05 07:24:56] ppdet.engine INFO: Total sample number: 199, averge FPS: 38.00272809256483
从评估结果来看，整体的检测指标还是比较高的，下面从实际测试情况看下具体的效果。

3.5 测试
In [9]
%cd /home/aistudio/PaddleDetection
!export CUDA_VISIBLE_DEVICES=0
!python tools/infer.py -c configs/ppyolo_mbv3_large_qr.yml --infer_dir=dataset/qr_coco/test --output_dir=output/test -o weights=./output/ppyolo_mbv3_large_qr/79.pdparams
检测结果位于output/test文件夹下，部分检测结果如下图所示：

       

从上面的测试图像可以看到，整个的检测精度还是比较好的，基本没有误检和漏检。基于深度学习的二维码检测算法相比传统图像处理检测算法在性能和鲁棒性上有了显著提升，尤其是PaddleDetection套件中的PPYolo算法经过高度的优化，所训练的模型轻量、精度高。

3.6 静态模型导出
我们前面训练出来的模型是动态图模型（.pdparams文件），可以直接使用PaddlePaddle进行推理，但是这种模型不适合跨平台、跨设备运行，而且本身用动态图模型进行推理效率比较低。真实工业场景部署时我们需要将这个模型导出成静态图模型（.pdiparams和.pdmodel文件），这个转换的核心原理就是将动态图中的所有不确定尺寸固定化，从而可以在编译前先把整个网络全部搭建好，再推理，这样虽然牺牲了动态图推理的灵活性（边执行边构建网络），但是却可以大幅提高推理速度和稳定性，对于工业部署来说尤为重要。

下面我们就将我们前面训练好的动态图模型进行静态化导出。

In [ ]
%cd /home/aistudio/PaddleDetection
!export CUDA_VISIBLE_DEVICES=0
!python tools/export_model.py -c configs/ppyolo_mbv3_large_qr.yml --output_dir=./output/inference_model \
 -o weights=./output/ppyolo_mbv3_large_qr/79.pdparams
模型导出后整个模型大概20M左右，模型比较小，因此，这个轻量级模型后面可以部署在众多边缘设备中，包括移动手机、jetson开发板、树莓派等。

3.7 静态图预测
静态图导出以后，我们就可以使用多种语言进行推理，包括Python、c++等。本小节先使用Python进行推理测试下是否成功。我们可以网上搜索一张真实的带二维码的图像。然后上传放置在PaddleDetection/demo文件夹中，命名为1.jpg，然后进行推理。

In [ ]
%cd /home/aistudio/PaddleDetection
!export CUDA_VISIBLE_DEVICES=0
!python deploy/python/infer.py --model_dir=./output/inference_model/ppyolo_mbv3_large_qr --image_file=demo/1.jpg --use_gpu=True
测试结果位于PaddleDetection/output文件夹下。

四、解码
4.1 解码库概述
前面我们已经将二维码检测出来了，接下来就是将二维码对应的ROI区域提取出来，然后送给解码器进行解码识读。目前存在着很多二维码解码库，其中由日本技术人员开发的Zbar及谷歌开发的Zxing这两种主流的解码库最令人耳熟能详。这两种解码库除了支持二维码的解码识读以外，还同时支持39码、EAN-8、EAN-13/UPC-A、128码等，并且提供了不同的开发语言供开发人员调用，另外，还支持跨平台开发。两者比较而言，Zbar的解码效率更高，因此本项目选用Zbar库来进行解码。

4.2 解码识读
因为我们需要操作静态图推理模型，因此我们可以参考PaddleDetection/deploy/python文件夹中的内容进行定制化推理。由于这部分代码可以脱离GPU环境（前面已经将模型静态图导出），因此，我们可以离线在本地进行代码调试。

由于zbar只支持Python2，而现在普遍都是使用Python3，因此不能直接使用zbar，这里我们使用一个开源的二维码识读库pyzbar，可以用来解码二维码和条形码。pyzbar支持两种数据类型输入：

PIL读取的PIL.Image数据类型
opencv读取的ndarray数据类型
首选进行安装：

In [6]
# windows安装方式
pip3 install pyzbar
In [ ]
# linux安装方式
sudo apt-get install libzbar0
注意，由于ai studio不能使用apt-get方式安装第三方库，因此只能线下使用。下面的所有内容均在线下环境中使用。如果读者还是想要在ai stuido上使用，也可以使用opencv4自带的二维码解码库进行解码，但是效果会差一些。

由于最终我们的目标是将所有算法移植到边缘设备（jetson）上，并且使用C++推理实现，因此，为了能真正做好这个项目产品，各个组件还是应该选择性能最好的。

下面我们在PaddleDetection/deploy/python目录下新建一个文件infer_qr.py,实现二维码检测和识别，具体内容不再展开，读者可以自行参考阅读infer_qr.py。

最终解码效果如下图所示：



五、总结
本文详细阐述了如何构建一个高精度的二维码识读仪，给出了详细的数据集准备、训练、推理、部署步骤。当然，对于真实的商业方案来说目前做的还不够，对于抠取出来的二维码区域我们直接使用Zbar来解码，但是很多情况下二维码是畸变的、磨损的，比较好的方法就是在抠取出二维码以后再加一个深度学习矫正算法，自动的对这种退化过后的二维码进行矫正，具体可以参考论文“DocUNet: Document Image Unwarping via A Stacked U-Net”。由于时间有限，本文不再深度扩展，有兴趣继续深入的读者可以加技术交流群：820106877。

六、参考文献
[1] Blanger L , Hirata N . An Evaluation of Deep Learning Techniques for Qr Code Detection[C]// 2019 IEEE International Conference on Image Processing (ICIP). IEEE, 2019.