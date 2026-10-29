import os
import sys
import numpy as np
import soundfile as sf
import scipy
from scipy.signal import stft
from rknnlite.api import RKNNLite as RKNN

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




SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_LENGTH = CHUNK_LENGTH * 100
N_MELS = 80


def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print("resample_audio: {} HZ -> {} HZ".format(original_sample_rate, desired_sample_rate))
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate

def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print("convert_channels: {} -> {}".format(original_channels, desired_channels))
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels

def get_char_index(c):
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A')
    elif 'a' <= c <= 'z':
        return ord(c) - ord('a') + (ord('Z') - ord('A') + 1)
    elif '0' <= c <= '9':
        return ord(c) - ord('0') + (ord('Z') - ord('A')) + (ord('z') - ord('a')) + 2
    elif c == '+':
        return 62
    elif c == '/':
        return 63
    else:
        print(f"Unknown character {ord(c)}, {c}")
        exit(-1)

def base64_decode(encoded_string):
    if not encoded_string:
        print("Empty string!")
        exit(-1)

    output_length = len(encoded_string) // 4 * 3
    decoded_string = bytearray(output_length)

    index = 0
    output_index = 0
    while index < len(encoded_string):
        if encoded_string[index] == '=':
            return " "

        first_byte = (get_char_index(encoded_string[index]) << 2) + ((get_char_index(encoded_string[index + 1]) & 0x30) >> 4)
        decoded_string[output_index] = first_byte

        if index + 2 < len(encoded_string) and encoded_string[index + 2] != '=':
            second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0f) << 4) + ((get_char_index(encoded_string[index + 2]) & 0x3c) >> 2)
            decoded_string[output_index + 1] = second_byte

            if index + 3 < len(encoded_string) and encoded_string[index + 3] != '=':
                third_byte = ((get_char_index(encoded_string[index + 2]) & 0x03) << 6) + get_char_index(encoded_string[index + 3])
                decoded_string[output_index + 2] = third_byte
                output_index += 3
            else:
                output_index += 2
        else:
            output_index += 1

        index += 4
            
    return decoded_string.decode('utf-8', errors='replace')

def read_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = {}
        for line in f:
            if len(line.strip().split(' ')) < 2:
                key = line.strip().split(' ')[0]
                value = ""
            else:
                key, value = line.strip().split(' ')
            vocab[key] = value
    return vocab

def pad_or_trim(audio_array):
    x_mel = np.zeros((N_MELS, MAX_LENGTH), dtype=np.float32)
    real_length = audio_array.shape[1] if audio_array.shape[1] <= MAX_LENGTH else MAX_LENGTH
    x_mel[:, :real_length] = audio_array[:, :real_length]
    return x_mel

def mel_filters(n_mels):
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    filters_path = "./model/mel_80_filters.txt"
    mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return mels_data

def log_mel_spectrogram(audio,n_mels=80,padding=0):

    if padding > 0:
        audio=np.pad(audio,(0,padding),mode='constant')
    
    # Create Hann window
    N_FFT = 201 * 2  # Assuming N_FFT is 402 based on the filter size (201 = N_FFT//2 + 1)
    HOP_LENGTH = 160  # Typical value, adjust if different
    
    window = np.hanning(N_FFT)
    f, t, Zxx = stft(audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    magnitudes = np.abs(Zxx[:-1]) ** 2  # 移除最后一个频率bin
    
    
    filters = mel_filters(n_mels)
    mel_spec = np.dot(filters, magnitudes)
    
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def load_array_from_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    array = []
    for line in data:
        row = [float(num) for num in line.split()]
        array.extend(row)

    return np.array(array).reshape((80, 2000))

class WhisperRKNN:
    def __init__(self,
                 enc_model_path:str='./model/whisper_encoder_rk3566_fp16.rknn',
                 dec_model_path:str='./model/whisper_decoder_rk3566_fp16.rknn',
                 task:str='zh',
                 target:str='rk3566'):
        self.enc_model=RKNN_model_container(enc_model_path,target,None)
        self.dec_model=RKNN_model_container(dec_model_path,target,None)

        if task=="en":
            self.vocab=read_vocab('./model/vocab_en.txt')
            self.task_code=50259
        elif task=='zh':
            self.vocab=read_vocab('./model/vocab_zh.txt')
            self.task_code=50260

    def encode(self,in_enc):
        out_enc=self.enc_model.infer(inputs=in_enc)[0]
        return out_enc

    def decode(self,out_enc):
        end_token = 50257
        tokens = [50258,self.task_code, 50359, 50363] 
        timestamp_begin = 50364

        max_tokens = 12
        tokens_str = ''
        pop_id = max_tokens

        tokens = tokens * int(max_tokens/4)
        next_token = 50258 # tokenizer.sot
        while next_token != end_token:
            out_decoder = self.dec_model.infer([np.asarray([tokens],dtype="int64"), out_enc])[0]
            next_token = out_decoder[0, -1].argmax()
            next_token_str = self.vocab[str(next_token)]
            tokens.append(next_token)

            if next_token == end_token:
                tokens.pop(-1)
                next_token = tokens[-1]
                break
            if next_token > timestamp_begin:
                continue
            if pop_id >4:
                pop_id -= 1

            tokens.pop(pop_id)
            tokens_str += next_token_str

        result = tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '').replace('\n', '')
        if self.task_code == 50260: # TASK_FOR_ZH
            result = base64_decode(result)
        return result


    def infer(self,audio_path:str='./model/test_en.wav'):
        audio_data, sample_rate = sf.read(audio_path)
        channels = audio_data.ndim
        audio_data, channels = ensure_channels(audio_data, channels)
        audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
        audio_array = np.array(audio_data, dtype=np.float32)
        audio_array= log_mel_spectrogram(audio_array, N_MELS)
        x_mel= pad_or_trim(audio_array)
        x_mel = np.expand_dims(x_mel,0)

        out_enc=self.encode(x_mel)
        result =self.decode(out_enc)
        return result
    
    def release(self):
        self.enc_model.release()
        self.dec_model.release()




if __name__ == '__main__':
    whisper=WhisperRKNN()
    result=whisper.infer()
    print(result)

 