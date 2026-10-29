# AR 眼镜端软件 说明文档

**项目**：面向复杂火场的 AR 眼镜与负压爬壁车协同系统（星火天巡 / 南京师范大学）
**模块**：单兵智能穿戴端（AR 眼镜）上位机软件
**编写日期**：2026-10-29
**运行平台**：瑞芯微 RK3588（Ubuntu 20.04/22.04、Debian 11/12），单目 AR 眼镜 + MicroOLED

---

## 一、项目概述

本目录（`ar/`）是协同系统中的 **AR 眼镜穿戴端**软件，承担三项职责：

1. **本地边缘智能**：在 RK3588 上以 RKNN 加速运行 PP-OCR 文字识别与 YOLOv5 人体检测；
2. **人机交互**：通过 MediaPipe 手势识别实现无接触 UI 操作，解放消防员双手；
3. **车端协同**：接收爬壁车第一视角视频流并回显，下发运动控制指令与遥测显示。

与之配套的爬壁车端代码见同级目录 `ar-slave/`（位于 `ar(改动版)/` 下）。

---

## 二、根目录文件说明

### 2.1 程序入口

| 文件 | 作用 |
|---|---|
| [main.py](main.py) | **主入口**。创建 `QApplication`、摄像头对象（默认设备号 21）、手势线程与主窗口，完成全部信号连接后启动事件循环。退出时先停止手势线程再返回退出码。 |
| [start_rk3588.py](start_rk3588.py) | **RK3588 专用启动脚本**（推荐入口）。启动前检测 CPU 是否为 RK3588、Python 版本、PyQt6/cv2/numpy 是否就绪；随后尝试将 CPU 切至 performance 模式并关闭屏幕保护，最后调用 `main.main()`。 |
| [run.sh](run.sh) | Conda 环境激活脚本。激活位于 `/home/elf/miniconda3` 的 `ar` 虚拟环境，供板端终端先 source 再运行 Python。 |

### 2.2 界面层

| 文件 | 作用 |
|---|---|
| [widget_main.py](widget_main.py) | **核心 UI 文件**（约 2400 行，体量最大）。包含主窗口 `MainWindow` 与全部功能页面。 |
| [func_choose.py](func_choose.py) | 早期的功能调度线程 `FuncThread`，按 `func_id` 分发到云端翻译、RTSP 取帧、PP-OCR 等任务，通过 `QSharedMemory` 读取手势线程写入的图像。**当前 `main.py` 未使用它**，属遗留代码。 |

`widget_main.py` 中的主要类：

| 类 | 功能 |
|---|---|
| `MainWindow` | 主窗口。1500×900 黑底；顶部安全系数状态栏；`QStackedLayout` 管理各功能页；30 ms 定时器轮询手势点击；2 s 定时器刷新安全系数；叠加 `GestureLayer` 绘制手部骨架。 |
| `MainContentWidget` | 主菜单页。5 个 280×280 图标按钮：文字识别、语音识别、远程控制、远程串流、目标检测。 |
| `TextRegWidget` | **文字识别页**。调用 PP-OCR 识别画面文字，支持"翻译"按钮（英→中），识别结果以 PIL 中文标注绘制在文本框内部。 |
| `SpeechRegWidget` | **语音识别页**。当前识别线程代码整体注释，仅保留 UI 骨架，**无实际功能**。 |
| `RemoteCtrlWidget` | **远程控制页**。爬壁车操控面板：连接状态指示、0–100% 调速、五向指令（前/后/左/右/停）、捏合手势连续控制（100 ms 周期）、内嵌 640×480 视频回显。 |
| `RemoteCameraWidget` | **远程串流页**。支持 HTTP-MJPEG 与 RTSP 拉流，地址可在界面输入；叠加车体遥测（电量/温度/湿度）与三项告警（障碍物/延迟/连接）；可叠加 YOLOv5 检测结果。 |
| `ObjectDetectionWidget` | **目标检测页**。RKNN 加速的 YOLOv5 人体检测；`showEvent`/`hideEvent` 自动启停推理，离开页面即释放 NPU。 |
| `GestureLayer` | 透明覆盖层，实时绘制手部关键点与骨架连线，提供手势操作的可视反馈。 |

### 2.3 手势与摄像头

| 文件 | 作用 |
|---|---|
| [hand_gesture.py](hand_gesture.py) | **手势识别核心**。`Camera` 封装摄像头参数（宽/高/设备号）；`KalmanFilter` 对关键点做滤波平滑；`get_finger_state` 判定各指弯曲/伸直；`MouseController` 判定捏合点击；`ImageSharedBuf` 封装共享内存读写；`GestureThread` 为采集主线程——读取帧、水平翻转修正镜像、按标志位向各页面分发图像、发射手势信号。 |
| [gesture_mouse.py](gesture_mouse.py) | 早期版本，直接用 `pyautogui` 驱动系统鼠标指针。当前 UI 改为手势直接操控界面，此文件未参与主流程。 |

### 2.4 网络通信

| 文件 | 作用 |
|---|---|
| [network_tools.py](network_tools.py) | **通信工具集**。`SyncUDPSender/Receiver` 同步 UDP；`AsyncUDPSender/Receiver` 基于 asyncio 的异步 UDP；`RTSPSender` 调用外部 ffmpeg 以 rkmpp 硬件编码推流（含日志监控线程）；`RTSPReceiver` 拉取远端 RTSP 流。是眼镜端与车端通信的底层。 |
| [phonetic_translation.py](phonetic_translation.py) | 阿里云 DashScope 语音翻译（含 pyaudio 录音）。主流程未调用，可独立运行验证。 |

### 2.5 模型与算法（`neural_net/`）

| 路径 | 作用 |
|---|---|
| `rknn/ppocr/ppocr.py` | **PP-OCRv4 文字识别**。`TextDetector`（检测）+ `TextRecognizer`（识别）两段式，封装为 `PPOCRRKNN.infer(img)` 返回文本框与文字。 |
| `rknn/ppocr/model/` | 检测模型 `ppocrv4_det.rknn`、识别模型 `ppocrv4_rec.rknn`、字典 `ppocr_keys_v1.txt`。 |
| `rknn/ppocr/utils/` | DB 后处理、识别后处理、图像算子等辅助实现。 |
| `rknn/yolov5/yolov5.py` | **YOLOv5 目标检测**，`Yolov5RKNN.infer(image)` 返回带框图像；类别含 COCO 80 类，火场场景聚焦 `person`。 |
| `rknn/yolov5/model/` | `yolov5_rknn7.0_best.rknn`（RKNN 部署模型）、`.pt` 训练权重、`anchors_yolov5.txt`、`coco_80_labels_list.txt`。 |
| `rknn/whisper/whisper.py` | Whisper 语音识别的 RKNN 移植（16 kHz / 400 点 FFT）。**主流程未接入**。 |
| `rknn/py_utils/` | `rknn_executor.py` 模型容器、`coco_utils.py` COCO 前后处理公共库。 |
| `cloud/gummy/gummy.py` | 阿里云 DashScope **实时语音听写与翻译**，供语音识别页使用。 |
| `cloud/translation/translation.py` | 基于 argostranslate 的**离线**英→中翻译，供文字识别页"翻译"按钮调用。 |

### 2.6 其他

| 文件 / 目录 | 作用 |
|---|---|
| [requirements.txt](requirements.txt) | Python 依赖清单（PyQt6、opencv-python、numpy、rknn-toolkit2 等）。 |
| [README_RK3588.md](README_RK3588.md) | RK3588 部署文档：系统依赖、RKNN 模型部署、性能调优、摄像头/蓝牙故障排查。**注意：标题误写为"负压爬壁机器人控制系统"，实际描述的是 AR 眼镜端。** |
| `reader` | 第二十一届中国研究生电子设计竞赛技术论文全文（中英文摘要、系统设计、测试分析、创新点）。 |
| [test.py](test.py) | RTSP 推流能力验证脚本，独立运行。 |
| [test1.py](test1.py) | 摄像头/HTTP 视频流连通性最小验证（按 q 退出、空格拍照）。 |
| `test/` | 各模块独立验证脚本：mediapipe 手势、opencv、语音翻译、ppocr、rknn、rtmp、rtsp、udp 收发等。 |
| `image/` | UI 图标资源（24 个 PNG）：功能图标、返回/翻译/开关、灯与风扇状态、盾形安全图标等。 |
| `project/` | C++ 工程：`src/`（camera、model、shader、invoke_python、stl_reader 等）、`python/`（gesture.py、main.py）、`CMakeLists.txt`（依赖 OpenCV、glfw）、lvgl 等第三方库。为早期/并行的原生渲染实现路径。 |

---

## 三、数据流向

```
摄像头 (Camera, 设备 21)
    │
    ▼
GestureThread ──图像帧──► 按标志位分发 ──► TextRegWidget       (PP-OCR)
    │                                  ──► ObjectDetectionWidget (YOLOv5)
    │                                  ──► RemoteCtrlWidget      (视频回显)
    │
    └─手势信号─► MainWindow ──► GestureLayer 绘制骨架
                             └─► 30ms 轮询判定点击 ──► 切换功能页 / 触发按钮

爬壁车 ──RTSP/HTTP 视频流──► RemoteCameraWidget ──► 叠加遥测与告警
AR 眼镜 ──UDP / 蓝牙控制指令──► 爬壁车
```

图像分发采用**按需机制**：各页面通过 `text_need_signal`、`od_need_signal` 等信号告知手势线程是否需要帧，避免多页面重复解码造成的算力浪费。

---

## 四、本次梳理中修复的问题

编写本文档时对代码做了通读，修复了以下 4 处会实际影响运行的问题：

**1. 启动脚本无法启动主程序（严重）**
`start_rk3588.py` 调用 `main.main()`，但 `main.py` 中不存在该函数，脚本必然抛 `AttributeError` 退出。已将逻辑封装为 `main()` 并补回 `if __name__ == "__main__"` 入口。

**2. 手势线程退出时资源泄漏**
原 `main.py` 在 `sys.exit(app.exec())` 之后才调用 `gesture_thread.stop()`，该行永不执行，摄像头句柄与 MediaPipe Hands 实例无法释放。已改为先取回退出码、再 `stop()` 并 `wait()`。

**3. 中文字体在 Linux 上回退失败（严重）**
`widget_main.py` 的 OCR 标注字体候选列表被改为仅含 Windows 路径（`C:/Windows/Fonts/...`），在 RK3588 的 Ubuntu/Debian 上全部失效，中文会渲染成方块。已恢复 Linux 字体路径（`wqy-zenhei.ttc`、`NotoSansCJK-Regular.ttc`）并置于 Windows 路径之前。

**4. 阿里云 API 密钥硬编码在源码中（安全）**
`neural_net/cloud/gummy/gummy.py` 第 14 行硬编码了 DashScope `sk-` 密钥。该密钥随代码目录复制/外发即会泄露。已改为**强制从环境变量 `DASHSCOPE_API_KEY` 读取**，缺失时直接报错退出而非静默使用内置密钥。启动前请先 `export DASHSCOPE_API_KEY=<your-key>`。

> 同类问题：`test/phonetic_translation_org.py:16` 也硬编码了另一个 `sk-` 密钥。该文件在 `test/` 下、不参与主流程，**未作改动**，建议自行删除或轮换该密钥。

---

## 五、已知限制

以下为**当前代码中确实存在、尚未解决**的问题，使用时请注意：

1. **机器人控制指令未真正下发。** `MainWindow.handle_robot_control()` 仅打印日志，无实际通信代码（`widget_main.py:2261`）。界面"已连接"为本地模拟状态，不会建立真实蓝牙/串口链路；`requirements.txt` 中的 `pybluez` 未被任何模块引用。
2. **手势方向识别为随机模拟。** `RemoteCtrlWidget.detect_gesture()` 使用 `random.random()` 产生方向结果，仅供演示，不可用于实际控制。
3. **安全系数为模拟数据。** `simulate_safety_factor()` 由 2 秒定时器生成，未接入真实传感器。
4. **语音识别模块已停用。** `SpeechRegWidget` 的识别线程整体注释，进入后无实际功能。
5. **部分依赖未被使用。** `python-docx`、`requests`、`psutil` 在代码中无引用，可从 `requirements.txt` 移除以精简部署体积。
6. **遗留代码未清理。** `func_choose.py`、`gesture_mouse.py` 已不参与主流程；`neural_net/rknn/whisper/` 未接入。
7. **文档标题与内容不符。** [README_RK3588.md](README_RK3588.md) 标题写"负压爬壁机器人控制系统"，实际描述 AR 眼镜端，建议更正。
8. **手势交互在佩戴厚重消防手套时识别率下降**，关键指令需依赖二次确认机制（论文 3.1 节已指出）。

---

## 六、安装与运行

```bash
# 1. 系统依赖
sudo apt update && sudo apt install -y python3 python3-pip python3-dev build-essential
sudo apt install -y libopencv-dev python3-opencv libqt6core6 libqt6gui6 libqt6widgets6

# 2. Python 依赖
pip3 install -r requirements.txt

# 3. 语音功能需配置密钥（不用语音功能可跳过）
export DASHSCOPE_API_KEY=<your-key>

# 4. 确认 RKNN 模型就位
ls neural_net/rknn/yolov5/model/     # 应含 yolov5_rknn7.0_best.rknn、anchors_yolov5.txt
ls neural_net/rknn/ppocr/model/      # 应含 ppocrv4_det.rknn、ppocrv4_rec.rknn

# 5. 启动（推荐）
chmod +x start_rk3588.py && ./start_rk3588.py

# 或直接启动
python3 main.py
```

摄像头权限问题请将当前用户加入 `video` 组：`sudo usermod -aG video $USER`（需重新登录生效）。

---

## 七、与其他目录的关系

| 目录 | 说明 |
|---|---|
| `ar/` | **当前版本**，本文档描述对象。功能最完整（含远程控制、串流、目标检测），有完整部署配套。 |
| `ar（优化版）/` | 上一版本。无远程控制页（仅有灯/风扇演示控件），无部署脚本与依赖清单。 |
| `ar（镜像版）/` | 早期版本，仅文字识别 + 拍照。 |
| `ar(改动版)/` | 分支实验版，含药品识别模块（`medicine_*.py`）与 `ar-slave/` 车端代码，与主线路无关。 |
| `ppYOLO/` | PaddlePaddle YOLO 相关实验目录，未参与本端主流程。 |
