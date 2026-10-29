# 负压爬壁机器人控制系统 - RK3588版本

## 系统要求

- 硬件平台：瑞芯微RK3588开发板
- 操作系统：Ubuntu 20.04/22.04 或 Debian 11/12
- 内存：至少4GB RAM
- 存储：至少16GB可用空间
- 摄像头：USB摄像头或MIPI摄像头
- 蓝牙模块：用于控制机器人

## 依赖安装

### 1. 系统依赖
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3 python3-pip python3-dev build-essential

sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libqt6core6 libqt6gui6 libqt6widgets6

sudo apt install -y bluez bluez-tools python3-bluez

sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev
```

### 2. Python依赖
```bash
# 安装Python库
pip3 install -r requirements.txt

# 如果没有requirements.txt文件，手动安装：
pip3 install PyQt6 opencv-python numpy python-docx
```

## RKNN模型部署

### 1. 安装RKNN Toolkit
```bash
# 安装RKNN Toolkit 2.x
pip3 install rknn-toolkit2

# 或者从瑞芯微官网下载对应版本
# https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheets/index.html
```

### 2. 部署YOLOv5模型
```bash
# 确保模型文件存在
ls neural_net/rknn/yolov5/model/

# 应该包含以下文件：
# yolov5_rknn7.0_best.rknn
# anchors_yolov5.txt
```

## 启动程序

### 方法1：使用专用启动脚本（推荐）
```bash
# 赋予执行权限
chmod +x start_rk3588.py

# 启动程序
./start_rk3588.py
```

### 方法2：直接启动
```bash
python3 main.py
```

## 性能优化建议

### 1. CPU性能模式
```bash
# 设置为性能模式（推荐）
sudo echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# 或者使用cpufrequtils
sudo apt install cpufrequtils
sudo cpufreq-set -g performance
```

### 2. GPU加速
```bash
# 确保安装了Mali GPU驱动
sudo apt install mali-g610-firmware

# 启用OpenCL加速
export OPENCV_OPENCL_DEVICE=:0
```

### 3. 内存优化
```bash
# 增加交换空间（如果内存不足）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 常见问题

### 1. 摄像头无法打开
```bash
# 检查摄像头是否被识别
ls /dev/video*

# 检查权限
ls -l /dev/video0

# 如果权限不足，添加用户到video组
sudo usermod -aG video $USER
```

### 2. 蓝牙连接失败
```bash
# 检查蓝牙服务状态
sudo systemctl status bluetooth

# 重启蓝牙服务
sudo systemctl restart bluetooth

# 扫描蓝牙设备
hcitool scan
```

### 3. 程序运行缓慢
```bash
# 检查CPU使用率
top

# 检查内存使用
free -h

# 关闭不必要的后台进程
sudo systemctl stop --now some-unneeded-service
```

### 4. RKNN模型加载失败
```bash
# 检查模型文件是否存在
ls neural_net/rknn/yolov5/model/yolov5_rknn7.0_best.rknn

# 检查RKNN Toolkit版本
pip3 show rknn-toolkit2

# 确保使用的是RK3588兼容的模型版本
```

## 调试日志

程序运行时会生成日志文件：
```bash
# 查看实时日志
tail -f robot_control.log

# 查看错误日志
grep "ERROR" robot_control.log
```

## 技术支持

如果遇到问题，请提供以下信息：
1. 开发板型号和操作系统版本
2. 完整的错误日志
3. 执行的命令和步骤
4. 相关截图或视频

## 版本历史

- v1.0.0: 初始版本，支持基本功能
- v1.1.0: 优化RK3588性能，修复已知bug
- v1.2.0: 添加手势控制，改进UI界面