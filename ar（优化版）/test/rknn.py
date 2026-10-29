import numpy as np
from rknnlite.api import RKNNLite as RKNN
import soundfile as sf

SAMPLE_RATE=16000
N_FFT=400
HOP_LENGTH=160
CHUNK_LENGTH=20
N_SAMPLES=CHUNK_LENGTH * SAMPLE_RATE

with open('./models/vocab.txt', 'r') as f:
    vocab={}
    for line in f:
        if len(line.strip().split(' ')) < 2:
            key=line.strip().split(' ')[0]
            value=""
        else:
            key, value=line.strip().split(' ')
        vocab[key]=value

def mel_filters(n_mels):
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    filters_path="./models/mel_80_filters.txt"
    mels_data=np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return mels_data

def pad_or_trim(array, length=N_SAMPLES, axis=-1):
    if array.shape[axis] > length:
        array=np.take(array, indices=range(length), axis=axis)
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
    stft_result=stft(audio, N_FFT, HOP_LENGTH,window)
    magnitudes=np.abs(stft_result) ** 2
    magnitudes=magnitudes.T
    filters=mel_filters(n_mels)
    assert filters.shape[1] == magnitudes.shape[0], \
        f"Mel filter shape {filters.shape} incompatible with magnitudes shape {magnitudes.shape}"
    mel_spec=np.dot(filters, magnitudes)
    
    log_spec=np.clip(mel_spec, a_min=1e-10, a_max=None)
    log_spec=np.log10(log_spec)
    log_spec=np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec=(log_spec + 4.0) / 4.0
    return log_spec

def init_rknn_model(model_path, device_id):
    rknn=RKNN()
    print('--> Loading model')
    ret=rknn.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model \"{}\" failed!'.format(model_path))
        exit(ret)
    print('done')
    
    print('--> Init runtime environment')
    ret=rknn.init_runtime(device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn

class WhisperRknn:
    def __init__(self, encoder_model_path, decoder_model_path):
        self.dev_id='None'
        self.encoder_model=init_rknn_model(encoder_model_path, self.dev_id)
        self.decoder_model=init_rknn_model(decoder_model_path, self.dev_id)
        self.x_mel=None
    
    def encode(self):
        out_encoder=self.encoder_model.inference(inputs=[self.x_mel])[0]
        return out_encoder

    def decode(self, out_encoder):
        end_token=50257
        tokens=[50258, 50259, 50359, 50363]
        timestamp_begin=50364 
        max_tokens=12
        tokens_str=''
        pop_id=max_tokens
        tokens=tokens * int(max_tokens/4)
        next_token=50258 
        
        while next_token != end_token:
            out_decoder=self.decoder_model.run(
                ["out"],
                {
                    "tokens": np.asarray([tokens], dtype="int64"),
                    "audio": out_encoder
                }
            )[0]
            next_token=np.argmax(out_decoder[0, -1])
            next_token_str=vocab[str(next_token)]
            tokens.append(next_token)
            
            if next_token == end_token:
                tokens.pop(-1)
                next_token=tokens[-1]
                break
            if next_token > timestamp_begin:
                continue
            if pop_id > 4:
                pop_id -= 1
            tokens.pop(pop_id)
            tokens_str += next_token_str
        
        result=tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '')
        return result

    def run(self):
        audio_data, sample_rate=sf.read("./audio/test.wav")
        audio_array=np.array(audio_data, dtype=np.float32)
        audio=pad_or_trim(audio_array.flatten())
        self.x_mel=log_mel_spectrogram(audio)
        out_encoder=self.encoder_model.inference(inputs=[self.x_mel])[0]
        result=self.decode(out_encoder)
        print("Whisper output:", result)

if __name__ == '__main__':
    whisper=WhisperRknn(
        './models/whisper_encoder_base_20s.rknn',
        './models/whisper_decoder_base_20s.rknn'
    )
    whisper.run()