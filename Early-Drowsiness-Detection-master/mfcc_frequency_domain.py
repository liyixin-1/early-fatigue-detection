import os

import numpy as np
from pip._internal.utils.misc import enum
from scipy import fftpack
import pandas as pd
from matplotlib import pyplot as plt
import librosa.display
def pad0(raw_data):
    frame_num = len(raw_data)
    inc = wlen - overlap
    nf = (frame_num - wlen) // inc + 1
    numfillzero = (nf * inc + wlen) - frame_num  # 填充0的个数
    fillzeros = np.zeros(numfillzero)  # 生成填充序列
    pad_data = np.append(raw_data, fillzeros)   # 将填充的序列与 未填充前序列合并
    return pad_data


# 分帧
def enframe(ear_data_list):
    frame_num = len(ear_data_list)
    #print(frame_num)    # 填充0之后的数据长度
    # assert len(ear_data_list) == len(ear_data_list)
    ear_list_enframe = []

    inc = wlen - overlap    # 帧移：80
    nf = (frame_num - wlen) // inc + 1  # 分成多少帧  224
    indf = np.multiply(inc, np.array([i for i in range(nf)]))   # 设置每帧在数据x中的位移量位置
    for i in range(nf):
        sub_ear = [ear_data_list[j] for j in range(indf[i], indf[i] + wlen)]
        ear_list_enframe.append(sub_ear)
    return ear_list_enframe
def mfcc(feature_flow=None):  # feature_flow是一个窗口的数据 shape:二维矩阵

    """
    傅里叶变换和功率谱
    """

    y = np.absolute(np.fft.rfft(feature_flow, nfft))  # fft的幅度(magnitude)  np.fft.rfft:通过称为快速傅里叶变换 (FFT) 的高效算法计算实值数组的一维 n-point 离散傅里叶变换 (DFT)
    # plot_spectro(y)
    # plot_fft(y)

    pow_frames = ((1.0 / nfft) * (y ** 2))  # Power Spectrum 功率谱
    # print(pow_frames.shape)

    """
    滤波
    """

    low_freq_mel = 0     # 选取最低频率为0
    # 将频率转换为Mel，并取对数
    nfilt = 10    # 设置区间数
    high_freq_mel = (2595 * np.log10(1 + (fps / 2) / 700))   # convert HZ to Mel
    # print(high_freq_mel)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale,划分出nfilt+2个间距
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((nfft + 1) * hz_points / fps)  # bin储存的是刻度对应的傅里叶变换点数

    # fbank特征
    # num_ceps = 4  4阶倒谱系数
    fbank = np.zeros((nfilt, int(np.floor( nfft / 2 + 1))))   # fbank存每个滤波器的值
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)  # 将功率谱与滤波器做点积
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 将filter_bank中的0值改为最小负数，防止运算出现问题
    filter_banks = 20 * np.log10(filter_banks)  # dB 对每个滤波器的能量取log得到log梅尔频谱

    """
    梅尔倒谱Mel-frequency Cepstral Coefficients (MFCCs)
    """

    num_ceps = 4# 倒谱系数的阶数
    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # (n, 6)
    # self.plot_mfcc(mfcc, num_ceps)
    return mfcc
def plot_mfcc(mfcc, num_ceps):
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 14
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    mfcc = mfcc.reshape((-1, num_ceps))
    # print((int(k1), int(k)))
    # if (int(k1), int(k)) == (3, 2):
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(mfcc, x_axis='time', sr=fps)
    plt.colorbar()
    plt.show()

def plot_spectro(y):
    spec = librosa.amplitude_to_db(y, ref=np.max)
    librosa.display.specshow(spec, sr=fps, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

def plot_fft(y):
    y = np.absolute(np.fft.fft(y, 512))
    plt.plot(y)
    plt.title('Spectrum')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.show()


# 将our数据以列表形式存放
path1 = 'gaze'
folds_list=os.listdir(path1)
for f,fold in enumerate(folds_list):
    path2=path1+'/'+fold
    folder_list=os.listdir(path2)
    for ID,folder in enumerate(folder_list):
        path3=path2+'/'+folder
        fol_list=os.listdir(path3)
        for i,fols in enumerate(fol_list):
            path4=path3+'/'+fols
            txt_tables_x = []
            txt_tables_y = []
            f = open(path4, "r", encoding='utf-8')
            line = f.readline()
            while line:
                    #list1 = line.split(' ')
                list1=line.split('\t')
                txt_x=eval(list1[0])
                txt_y=eval(list1[1])
                # txt_data = eval(line)  # 可将字符串变为元组
                txt_tables_x.append(txt_x)  # 列表增加
                txt_tables_y.append(txt_y)
                line = f.readline()  # 读取下一行
            f.close()
            fps = 25
            wlen = 100  # 240
            overlap = 60
            # nfft = wlen  # 在每个帧上执行N点FFT来计算频谱,这也称为短时傅立叶变换(STFT)
            nfft = 100
            after_padding1 = pad0(txt_tables_x)  # 填充0之后的数据
            after_padding2 = pad0(txt_tables_y)
            gaze_data_list1 = after_padding1
            gaze_data_list2 = after_padding2
            after_enframe1 = enframe(gaze_data_list1)  # 输出分帧之后的数据
            after_enframe2 = enframe(gaze_data_list2)

            np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大

            # 数值不以科学计数法输出
            np.set_printoptions(suppress=True)

            fre_domain_x = mfcc(after_enframe1)
            fre_domain_y = mfcc(after_enframe2)
            fre_domain=np.concatenate((fre_domain_x,fre_domain_y),axis=1)
            if (fols == 'gaze_alert.txt'):
                outputpath = 'gazemfcc4w200o60/' + fold + '/'+folder +'/'+ 'gaze_mfcc_4_w200_o60_alert.txt'

            elif (fols == 'gaze_semisleepy.txt'):
                outputpath = 'gazemfcc4w200o60/' + fold + '/'+folder +'/'+ 'gaze_mfcc_4_w100_o60_semisleepy.txt'
            elif (fols == 'gaze_sleepy.txt'):
                outputpath = 'gazemfcc4w100o60/' + fold + '/'+folder +'/'+  'gaze_mfcc_4_w100_o60_sleepy.txt'
            np.savetxt(outputpath, fre_domain, fmt="%.8f")
'''
folders = "data/Fold5"
folds_list = os.listdir(folders)
for f,fold in enumerate(folds_list):
    path1=folders+'/'+fold
    folder_list=os.listdir(path1)
    for ID,folder in enumerate(folder_list):
        path2=path1+'/'+folder
        #path2=path1+'/ear-alert.txt'
        txt_tables = []
        f=open(path2,"r",encoding='utf-8')
        line = f.readline()
        while line:
            txt_data = eval(line)  # 可将字符串变为元组
            txt_tables.append(txt_data)  # 列表增加
            line = f.readline()  # 读取下一行
        f.close()


        fps = 25
        wlen = 100  # 240
        overlap = 60
        nfft = wlen  # 在每个帧上执行N点FFT来计算频谱,这也称为短时傅立叶变换(STFT)

        after_padding = pad0(txt_tables)  # 填充0之后的数据
        ear_data_list = after_padding
        after_enframe = enframe(ear_data_list)  # 输出分帧之后的数据

        np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大

        # 数值不以科学计数法输出
        np.set_printoptions(suppress=True)

        fre_domain = mfcc(after_enframe)
        if (folder=='ear-alert.txt'):
            outputpath='neww100o60/Fold5/'+fold+'/'+'mfcc_4_w100_o60_alert.txt'

        elif (folder=='ear-semisleepy.txt'):
            outputpath = 'neww100o60/Fold5/' + fold + '/' + 'mfcc_4_w100_o60_semisleepy.txt'
        elif (folder=='ear-sleepy.txt'):
            outputpath = 'neww100o60/Fold5/' + fold + '/' + 'mfcc_4_w100_o60_sleepy.txt'
       #print(fre_domain)
        # print(fre_domain.shape)
        np.savetxt(outputpath,fre_domain,fmt="%.8f")
        # f = open(outputpath, "w")
        # print(fre_domain, file=f)
        # f.close()
'''

'''
path2='our/sleepy.txt'
#ear-semisleepy.txt   ear-sleepy.txt
txt_tables = []
f=open(path2,"r",encoding='utf-8')
line = f.readline()
while line:
    print(line)
    txt_data = eval(line)  # 可将字符串变为元组
    txt_tables.append(txt_data)  # 列表增加
    line = f.readline()  # 读取下一行
f.close()
fps = 25
wlen = 100 #
overlap = 60
nfft = wlen  # 在每个帧上执行N点FFT来计算频谱,这也称为短时傅立叶变换(STFT)

after_padding = pad0(txt_tables)  # 填充0之后的数据
ear_data_list = after_padding
after_enframe = enframe(ear_data_list)  # 输出分帧之后的数据

np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大

# 数值不以科学计数法输出
np.set_printoptions(suppress=True)

fre_domain = mfcc(after_enframe)
outputpath='ourmfcc/mfcc_5_w100_o60_sleepy.txt'
# outputpath='newmfcc5w100o60/Fold1/04/mfcc_5_w100_o60_semisleepy.txt'
# outputpath='newmfcc5w100o60/Fold1/04/mfcc_5_w100_o60_sleepy.txt'
#np.savetxt(outputpath,fre_domain,fmt="%.8f")
'''