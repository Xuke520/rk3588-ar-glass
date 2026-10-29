import cv2
import os
import socket
import subprocess
import asyncio
import logging
import time
from typing import Optional,Tuple
import queue
# from neural_net.rknn.yolov5.yolov5 import Yolov5RKNN 
import mmap

'''
同步UDP:调用socket.sendto或者socket.revfrom时IO阻塞,直到数据发送或接收完成
异步UDP:通过事件循环asyncio,可以在IO操作时执行其它任务

'''

class SyncUDPSender:
    #构造函数
    def __init__(self,
                 target_ip:str="192.168.43.139",
                 target_port:int=30000):
        self.target_ip=target_ip        #目标ip
        self.target_port=target_port    #目标端口
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    def send_message(self,data:str)->bool:
        try:
            self.socket.sendto(data.encode("utf-8"),(self.target_ip,self.target_port))
            return True
        except Exception as e:
            return False
        
    def close(self):
        if self.socket:
            self.socket.close()
            self.socket=None

class SyncUDPReceiver:
    def __init__(self,
                 local_ip:str='0.0.0.0',    #监听所有网口
                 #local_ip='192.168.1.2',   #监听具体的ip地址
                 local_port:int=30000):
        self.local_ip=local_ip      #监听ip
        self.local_port=local_port  #监听端口
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((local_ip,local_port))

    def receive_message(self):
        return self.socket.recvfrom(1024)
    
    def close(self):
        if self.socket:
            self.socket.close()
            self.socket=None

class AsyncUDPSender:
    def __init__(self,
                 target_ip:str="192.168.43.139",
                 target_port:int=30000):
        #Optional[...]表示变量可以是None或指定的类型syncio.DatagramTransport
        self.target_ip=target_ip
        self.target_port=target_port
        self.transport:Optional[asyncio.DatagramTransport]=None #传输层

    async def start(self):
        loop=asyncio.get_running_loop()
        self.transport,_=await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(self.target_ip,self.target_port)
        )
        print(f"AsyncUDPSender to {self.local_ip}:{self.local_port}")
    
    async def send(self,data:bytes):
        self.transport.sendto(data,(self.target_ip,self.target_port))
    
    def close(self):
        if self.transport:
            self.transport.close()


'''
自定义协议类
继承自asyncio.DatagramProtocol
'''
class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self,
                 queue:asyncio.Queue):
        self.queue=queue

    #当接收到数据报时自动调用
    def datagram_received(self,data:bytes,addr:Tuple[str,int]):
        self.queue.put_nowait((data,addr))


class AsyncUDPReceiver:
    def __init__(self,
                local_ip:str='0.0.0.0',
                local_port:int=30000):
        self.local_ip=local_ip
        self.local_port=local_port
        self.transport:Optional[asyncio.DatagramTransport]=None #传输层
        self.protocol:Optional[UDPProtocol]=None    #协议层
        self.queue=asyncio.Queue()  #异步消息队列

    async def start(self):
        loop=asyncio.get_running_loop()
        self.transport,self.protocol=await loop.create_datagram_endpoint(
            lambda:UDPProtocol(self.queue),
            local_addr=(self.local_ip,self.local_port))
        print(f"AsyncUDPReceiver listening on {self.local_ip}:{self.local_port}")

    async def receive(self)->Tuple[bytes,Tuple[str,int]]:
            return await self.queue.get()

    def close(self):
        if self.transport:
            self.transport.close()


async def udp_receiver():
    receiver=AsyncUDPReceiver()
    await receiver.start()
    try:
        while True:
            data, addr=await receiver.receive()
            print(f"Received from {addr}: {data.decode()}")
    except asyncio.CancelledError:
        receiver.close()


async def udp_sender():
    sender=AsyncUDPSender()
    await sender.start()
    for i in range(5):
        await sender.send(f"send0000message {i}".encode())  # 移除了多余的地址参数
    sender.close()




#RTSP推流
class RTSPSender:
    def __init__(self):
        self.params={
            'input_device':'/dev/video28',
            'input_format':'yuyv422',
            'resolution':'640x480',
            'framerate':30,
            'rtsp_url':'rtsp://192.168.137.184:8554/cam',
            'hwaccel_device':'/dev/dri/renderD128',
        }
        self.process: Optional[subprocess.Popen]=None
        #初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    
    def build_command(self)->list:
        return [
            'ffmpeg',
            '-hwaccel','rkmpp',
            '-hwaccel_device',self.params['hwaccel_device'],
            '-f','v4l2',
            '-input_format',self.params['input_format'],
            '-video_size',self.params['resolution'],
            '-framerate',str(self.params['framerate']),
            '-i',self.params['input_device'],
            '-vf','format=nv12,hwupload',
            '-c:v','h264_rkmpp',
            '-b:v','2000k',
            '-f','rtsp',
            self.params['rtsp_url']
        ]

    def update_params(self,key:str,value):
        if key in self.params:
            old_val=self.params[key]
            self.params[key]=value
            logging.info(f"Updated {key}:{old_val}->{value}")
        else:
            logging.warning(f"ignored invalid parameter: {key}")

    def start(self):
        #启动RTSP推流
        if self.process and self.process.poll() is None:
            logging.warning("stream is already running")
            return
        try:
            cmd=self.build_command()
            logging.info("starting stream with command:\n"+" ".join(cmd))
            self.process=subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  #行缓冲
            )
            #启动日志监控线程
            self.start_log_monitor()
        except Exception as e:
            logging.error(f"failed to start stream: {str(e)}")
            raise

    def start_log_monitor(self):
        #监控FFmpeg输出
        import threading
        def monitor():
            while self.process and self.process.poll() is None:
                output=self.process.stdout.readline()
                if output:
                    logging.info(output.strip())
            logging.info("Stream process ended")
        #daemon=True,创建守护线程
        threading.Thread(target=monitor, daemon=True).start()
    
    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                logging.error(f"failed to stop process: {str(e)}")
            finally:
                self.process=None
                logging.info("Stream stopped")


class RTSPSenderTest:
    def __init__(self):
        self.params={
            'input_format':'rawvideo',
            'resolution':'640x480',
            'framerate':30,
            'rtsp_url':'rtsp://192.168.137.184:8554/cam',
            'hwaccel_device':'/dev/dri/renderD128',
        }
        self.process:Optional[subprocess.Popen]=None
        self.frame_queue=queue.Queue(maxsize=15)
        self.running=False
        #初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    
    def build_command(self)->list:
        return [
            'ffmpeg',
            '-hwaccel','rkmpp',
            '-re',
            '-hwaccel_device',self.params['hwaccel_device'],
            '-f', 'rawvideo',  
            '-pixel_format', 'bgr24',
            '-video_size',self.params['resolution'],
            '-framerate',str(self.params['framerate']),
            '-i','-',
            '-g', '15',
            '-vf','format=nv12,hwupload',
            '-c:v','h264_rkmpp',
            '-b:v','2000k',
            '-f','rtsp',
            '-muxdelay', '0.1',
            self.params['rtsp_url']
        ]


    def update_params(self,key:str,value):
        if key in self.params:
            old_val=self.params[key]
            self.params[key]=value
            logging.info(f"Updated {key}:{old_val}->{value}")
        else:
            logging.warning(f"ignored invalid parameter: {key}")

    def push_frame(self,frame):
        if not self.running and frame is None:
            return
    
        try:
            self.frame_queue.put(frame,block=False)
        except queue.Full:
            logging.warning("Frame queue full, dropping frame")
        except Exception as e:
            logging.error(f"Push frame failed: {str(e)}")

    def start_frame_sender(self):
        """负责从队列取帧并发送给FFmpeg"""
        import threading
        
        def sender():
            while self.running and self.process and self.process.poll() is None:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    if frame is not None:
                        self.process.stdin.write(frame)
                        self.process.stdin.flush()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error sending frame: {str(e)}")
                    break
            
            logging.info("Frame sender stopped")
        
        threading.Thread(target=sender,daemon=True).start()

    def start(self):
        #启动RTSP推流
        if self.process and self.process.poll() is None:
            logging.warning("stream is already running")
            return
        try:
            cmd=self.build_command()
            logging.info("starting stream with command:\n"+" ".join(cmd))
            self.process=subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=0  #行缓冲
            )
            self.running=True
            self.start_frame_sender()
            #启动日志监控线程
            self.start_log_monitor()
        except Exception as e:
            logging.error(f"failed to start stream: {str(e)}")
            raise

    def start_log_monitor(self):
        #监控FFmpeg输出
        import threading
        def monitor():
            while self.process and self.process.poll() is None:
                output=self.process.stdout.readline()
                if output:
                    logging.info(output.strip())
            logging.info("Stream process ended")
        #daemon=True,创建守护线程
        threading.Thread(target=monitor, daemon=True).start()
    
    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                logging.error(f"failed to stop process: {str(e)}")
            finally:
                self.process=None
                logging.info("Stream stopped")



    

#RTSP拉流
class RTSPReceiver:
    def __init__(self,
                 rtsp_url='rtsp://192.168.137.184:8554/cam'):
        self.rtsp_url=rtsp_url
        self.cap=None
    
    def connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap=cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)


    
    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.connect()
        ret,frame=self.cap.read()
        if ret:
            return frame
        else:
            return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":


    rtsp=RTSPReceiver()
    try:
        while True:
            frame = rtsp.get_frame()
            if frame is not None:
                cv2.imshow("RTSP Stream", frame)
            # 按'q'退出，控制帧率（约30fps）
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        rtsp.release()



    

   
