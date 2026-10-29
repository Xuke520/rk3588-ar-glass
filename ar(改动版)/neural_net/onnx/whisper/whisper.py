import numpy as np
import soundfile as sf
import onnxruntime

SAMPLE_RATE=16000#16000hz
N_FFT=400
HOP_LENGTH=160
CHUNK_LENGTH=20#20s音频
N_SAMPLES=CHUNK_LENGTH*SAMPLE_RATE#32000

n_frames = 1 + (N_SAMPLES - N_FFT) // HOP_LENGTH  # 应为 2000

# 加载词汇表
with open('./model/vocab.txt', 'r') as f:
    vocab={}
    for line in f:
        if len(line.strip().split(' ')) < 2:
            key=line.strip().split(' ')[0]
            value=""
        else:
            key, value=line.strip().split(' ')
        vocab[key]=value

def mel_filters(n_mels, filters_path):
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    mels_data=np.loadtxt(filters_path, dtype=np.float32).reshape((80,201))
    return mels_data

def pad_or_trim(array,length=N_SAMPLES,axis=-1):
    if array.shape[axis]>length:
        array=array.take(indices=range(length), axis=axis)
    if array.shape[axis] < length:
        pad_widths=[(0, 0)] * array.ndim
        pad_widths[axis]=(0, length - array.shape[axis])
        array=np.pad(array, pad_widths)
    return array

def stft(x, n_fft, hop_length, window):
    n_frames=1 + (len(x) - n_fft) // hop_length
    stft_result=np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        start=i * hop_length
        end=start + n_fft
        frame=x[start:end] * window
        frame_fft=np.fft.rfft(frame, n=n_fft)
        stft_result[i]=frame_fft
    return stft_result

def log_mel_spectrogram(audio, n_mels=80, padding=0):
    if padding > 0:
        audio=np.pad(audio, (0, padding))
    window=np.hanning(N_FFT)
    stft_result=stft(audio, N_FFT, HOP_LENGTH, window)
    magnitudes=np.abs(stft_result) ** 2
    magnitudes=magnitudes.T
    filters=mel_filters(n_mels, filters_path="./model/mel_80_filters.txt")
    mel_spec=np.dot(filters, magnitudes)
    
    if mel_spec.shape[1] < 2000:
        mel_spec=np.pad(mel_spec, ((0, 0), (0, 2000 - mel_spec.shape[1])))
    elif mel_spec.shape[1] > 2000:
        mel_spec=mel_spec[:, :2000]
    
    log_spec=np.clip(mel_spec, a_min=1e-10, a_max=None)
    log_spec=np.log10(log_spec)
    log_spec=np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec=(log_spec + 4.0) / 4.0
    return log_spec

def init_onnx_model(model_path):
    model=onnxruntime.InferenceSession(model_path,  providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    return model

class WhisperONNX:
    def __init__(self,
                 encoder_model_path:str='./model/whisper_encoder_base_20s.onnx',
                 decoder_model_path:str='./model/whisper_decoder_base_20s.onnx'):
        #如果可用,优先使用Azure提供的加速执行（如 GPU 加速）。CPUExecutionProvider：作为后备选项,确保模型在 CPU 上也能运行。
        self.encoder_model=onnxruntime.InferenceSession(encoder_model_path,providers=['AzureExecutionProvider','CPUExecutionProvider'])
        self.decoder_model=onnxruntime.InferenceSession(decoder_model_path,providers=['AzureExecutionProvider','CPUExecutionProvider'])
        self.x_mel=None

    def encode(self):
        encoder=self.encoder_model.run(None, {"x": [self.x_mel]})[0]
        return encoder


    def decode(self,encoder):
        end_token=50257 
        tokens=[50258,50259,50359,50363] 
        timestamp_begin=50364 

        max_tokens=12
        tokens_str=''
        pop_id=max_tokens

        tokens=tokens*int(max_tokens/4)
        next_token=50258 # tokenizer.sot

        while next_token != end_token:
            out_decoder=self.decoder_model.run(["out"], {"tokens": np.asarray([tokens], dtype="int64"), "audio": encoder})[0]
            next_token=out_decoder[0, -1].argmax()
            next_token_str=vocab[str(next_token)]
            tokens.append(next_token)
            if next_token == end_token:
                tokens.pop(-1)
                next_token=tokens[-1]
                break
            if next_token > timestamp_begin:
                continue
            if pop_id >4:
                pop_id -= 1
            tokens.pop(pop_id)
            tokens_str += next_token_str

        result=tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '')
        return result
    
    def infer(self):
         # 加载音频并提取特征
        audio_data, sample_rate=sf.read("./model/test.wav")
        audio_array=np.array(audio_data, dtype=np.float32)
        audio=pad_or_trim(audio_array.flatten())
        self.x_mel=log_mel_spectrogram(audio)
        encoder=self.encode()
        result=self.decode(encoder)
        return result
    
    def release(self):
        if hasattr(self, 'encoder_model'):
            del self.encoder_model
        if hasattr(self, 'decoder_model'):
            del self.decoder_model



            
if __name__ == '__main__':
    wh=WhisperONNX('./model/whisper_encoder_base_20s.onnx','./model/whisper_decoder_base_20s.onnx')
    result=wh.infer()
    print("Whisper output:", result)