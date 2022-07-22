from librosa import load,stft,magphase
from librosa.feature import mfcc
import librosa
import numpy as np
from torch.utils import data

def MFCC(audio_path,mode='train', sr=16000):

    wav, sr_ret = load(audio_path, sr=sr)  # sr为采样率:16K
    # raw_wav, sr_ret = librosa.load(audio_path, sr=sr)  # sr为采样率:16K

    while len(wav)<80000:
        wav=np.append(wav,wav)
    wav=wav[:80000]

    if mode == 'train':
        if np.random.random() < 0.3:
            wav = wav[::-1]      ##随机音频翻转

    mfcc_params=mfcc(y=wav, sr=sr,hop_length=160,n_mfcc=13)   ##hop_length：帧移 一般取采样率的0.001
    ##标准差标准化
    mean = np.mean(mfcc_params, 0, keepdims=True)
    std = np.std(mfcc_params, 0, keepdims=True)  #计算每一列的标准差      数据减去平均数再除以标准差叫标准差标准化，经过处理的数据符合正态分布
    mfcc_params = (mfcc_params - mean) / (std + 1e-5)##标准差标准化，使网络更容易收敛
    mfcc_params = mfcc_params[np.newaxis, :]


    return mfcc_params


# 加载并预处理音频
def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    # 读取音频数据
    wav, sr_ret = load(audio_path, sr=sr)           #sr为采样率:16K  采样一秒，wav长度为16000

    # 数据拼接
    if mode == 'train':
        extended_wav = np.append(wav, wav)   #音频拼接
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    elif mode =="infer":
        if len(wav>80000):
            extended_wav = wav[:80000]  # 不动
        else:
            extend_wav=wav
    else:
        extended_wav = np.append(wav, wav[::-1])   #翻转


    #print("extended_wav.shape",extended_wav)
    # 计算短时傅里叶变换
    linear = stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)#每一帧音频都由window()加窗，窗长为winlength，hop_length为帧移
    ##窗口默认为汉明窗    .fft长度为512，hop_length为160，则分为了4帧，每一帧都由window加窗
    mag, _ = magphase(linear)  #计算图谱的幅度和相位值
    freq, freq_time = mag.shape
    #print("mag.shape",mag.shape)
    assert freq_time >= spec_len, "非静音部分长度不能低于1.3s"
    if mode == 'train':
        # 随机裁剪
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag[:, rand_time:rand_time + spec_len]
    elif mode == "infer":
        spec_mag = mag[:]  # 不动
    else:
        spec_mag = mag[:, :spec_len]
        #
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)  #计算每一列的标准差      数据减去平均数再除以标准差叫标准差标准化，经过处理的数据符合正态分布
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    #print(spec_mag.shape)
    spec_mag = spec_mag[np.newaxis, :]
    #print(spec_mag.shape)
    #、 print('audio_path',audio_path)
    return spec_mag


# 数据加载器
class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, model='train', spec_len=257):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        #spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        mfcc=MFCC(audio_path,mode=self.model)
        #mel=Mel(audio_path,mode=self.model)
        return mfcc, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)







