import os
from typing import Optional
import pyaudio
import dashscope
from dashscope.audio.asr import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer

def api_key_init():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ[
        'DASHSCOPE_API_KEY'] 
    else:
        dashscope.api_key = 'sk-8ceec3ca7b3843eca096746ab7ad543d'  

class TranslationCallback(TranslationRecognizerCallback):
    def __init__(self,
                 target_Language:str='zh'):
        
        #类注解,表明self.mic可以是pyaudio.PyAudio或None
        self.mic:Optional[pyaudio.PyAudio]=None
        self.stream:Optional[pyaudio.Stream]=None
        self.target_language=target_Language
        self.translation_text=None
        self.new_translation_event=False

    #服务器连接成功时
    def on_open(self)->None:
        try:
            self.mic=pyaudio.PyAudio()
            self.stream=self.mic.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=16000,
                                    input=True)
        except Exception as e:
            raise RuntimeError(f"audio init fail:{str(e)}")
        
    #翻译服务关闭时
    def on_close(self)->None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()

    #发送错误时
    def on_error(self,error:Exception)->None:
        self.on_close()
    
    #收到翻译结果时
    def on_event(self, 
                 request_id,
                 transcription_result,
                 translation_result:TranslationResult,
                 usage)->None:
        #翻译结果
        if translation_result:
            translation=translation_result.get_translation(self.target_language)
            if translation.is_sentence_end:
                # print('translate to {}: {}'.format(self.target_language,translation.text))
                self.translation_text=translation.text
                self.new_translation_event=True
    
    def get_translation_text(self):
        if self.new_translation_event:
            self.new_translation_event=False
            return self.translation_text
        return None
  

class GummyCloud:
    def __init__(self,
                target_language:str='zh'):
        self.target_language=target_language
        self.callback=TranslationCallback(self.target_language)
        self.translator=TranslationRecognizerRealtime(
                        model='gummy-realtime-v1',  #指定模型
                        format='pcm',               #音频格式
                        sample_rate=16000,          #采样率
                        transcription_enabled=False,#禁用转录（仅翻译）
                        translation_enabled=True,   #启用翻译
                        translation_target_languages=[self.target_language],  #目标语言（中文）
                        callback=self.callback,          # 设置回调处理器
                    )
        api_key_init()
        self.translator.start()

    def infer(self):
        if self.callback.stream:
            data=self.callback.stream.read(3200,exception_on_overflow=False)
            self.translator.send_audio_frame(data)
        translation_text=self.callback.get_translation_text()
        if translation_text:
            return translation_text
        return None

    def release(self):
        if self.translator:
            self.translator.stop()
        if self.callback:
            self.callback.on_close()

class TranslationWindow(QWidget):
    def __init__(self,
                 gummy_cloud,
                 parent=None):
        super().__init__(parent)
        self.gummy_cloud=gummy_cloud
        self.init_ui()

    def init_ui(self):

        layout=QVBoxLayout(self)

        # 创建一个文本框用于显示翻译文本
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)  # 设置为只读
        font = self.text_edit.font()  # 获取当前字体
        font.setPointSize(30)   # 设置字体大小为 12 磅
        self.text_edit.setFont(font) # 应用字体
        layout.addWidget(self.text_edit)

        # 创建一个定时器，定期调用更新翻译文本
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_translation)
        self.timer.start(80)  # 每100毫秒更新一次

    def update_translation(self):
        # 从 GummyCloud 获取翻译文本
        translation = self.gummy_cloud.infer()
        if translation:
            # 在文本框中显示翻译结果
            # self.text_edit.append(translation)  # 追加文本
            self.text_edit.setPlainText(translation)  # 替换文本

    def closeEvent(self, event):
        # 关闭窗口时释放资源
        self.gummy_cloud.release()
        super().closeEvent(event)


# if __name__=='__main__':
    # gc=GummyCloud()
    # while True:
    #     print(gc.infer())

if __name__ == '__main__':
    # 初始化Qt应用
    app = QApplication([])
    main_window = QMainWindow()
    main_window.setWindowTitle("实时翻译应用")
    
    # 创建GummyCloud实例
    gc = GummyCloud()
    # 创建并显示翻译窗口
    window = TranslationWindow(gc)
    main_window.setCentralWidget(window)


    main_window.show()
    
    # 启动事件循环
    app.exec()
    
    # 程序退出时释放资源
    gc.release()
