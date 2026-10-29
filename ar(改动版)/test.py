class RTSPSender:
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

