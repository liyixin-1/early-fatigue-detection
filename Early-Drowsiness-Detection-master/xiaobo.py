import time
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import pywt
from pathlib import Path
import csv
from sklearn import preprocessing
import pywt
import pywt.data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# # 需要分析的四个频段
# iter_freqs = [
#     {'name': 'Delta', 'fmin': 0, 'fmax': 4},
#     {'name': 'Theta', 'fmin': 4, 'fmax': 8},
#     {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
#     {'name': 'Beta', 'fmin': 13, 'fmax': 35},
# ]


# mne.set_log_level(False)


########################################小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致)#########################
# def TimeFrequencyWP(data, fs, wavelet, maxlevel=8, s1='1', name='name'):
#     # 小波包变换这里的采样频率为250，如果maxlevel太小部分波段分析不到
#     wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
#     freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
#     print(len(freqTree))
#     # 计算maxlevel最小频段的带宽
#     freqBand = fs / (2 ** maxlevel)
#     print(len(wp[freqTree[5]].data))
#     #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
#     # 绘图显示
#     fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(10, 7), sharex=True, sharey=False)
#     # 绘制原始数据
#     axes[0].plot(data)
#     axes[0].set_title(name)
#     for iter in range(len(iter_freqs)):
#         # 构造空的小波包
#         new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#         for i in range(len(freqTree)):
#             # 第i个频段的最小频率
#             bandMin = i * freqBand
#             # 第i个频段的最大频率
#             bandMax = bandMin + freqBand
#             # 判断第i个频段是否在要分析的范围内
#             if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
#                 # 给新构造的小波包参数赋值
#                 new_wp[freqTree[i]] = wp[freqTree[i]].data
#         # 绘制对应频率的数据
#         axes[iter + 1].plot(new_wp.reconstruct(update=True))
#         # 设置图名
#         axes[iter + 1].set_title(iter_freqs[iter]['name'])
#     plt.show()
#     plt.savefig(s1, dpi=200, bbox_inches='tight')

#
# def WPEnergy(data, fs, wavelet, maxlevel=6):
#     # 小波包分解
#     wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
#     freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
#     # 计算maxlevel最小频段的带宽
#     freqBand = fs / (2 ** maxlevel)
#     # 定义能量数组
#     energy = []
#     # 循环遍历计算四个频段对应的能量
#     for iter in range(len(iter_freqs)):
#         iterEnergy = 0.0
#         for i in range(len(freqTree)):
#             # 第i个频段的最小频率
#             bandMin = i * freqBand
#             # 第i个频段的最大频率
#             bandMax = bandMin + freqBand
#             # 判断第i个频段是否在要分析的范围内
#             if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
#                 # 计算对应频段的累加和
#                 iterEnergy += pow(np.linalg.norm(wp[freqTree[i]].data, ord=None), 2)
#         # 保存四个频段对应的能量和
#         energy.append(iterEnergy)
#     # 绘制能量分布图
#     plt.plot([xLabel['name'] for xLabel in iter_freqs], energy, lw=0, marker='o')
#     plt.title('能量分布')
#     plt.show()


# signal为输入信号，n为分解层数
def wpd_plt(signal, n):
    # wpd分解 ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp',
    # 'cmor'] ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric',
    # 'antireflect']
    wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=n)

    # 计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = signal
    for row in range(1, n + 1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # 作图
    # plt.cla()
    # plt.close("all")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = ['Helvetica']
    # plt.style.use(['science','ieee','no-latex'])
    # matplotlib.rc('font',family='Times New Roman')
    # plt.rcParams['text.usetex']=True

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 8

    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    plt.rcParams['lines.linewidth'] = 1.5
    plt.figure(figsize=(12, 10))

    plt.subplot(n + 1, 1, 1)  # 绘制第一个图
    plt.plot(map[1],color=(0,0,0.2,0.5))
    plt.title('Fatigue',fontdict={'weight':'normal','size':30})
    #plt.ylabel('Decomposition Factor')

    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # 从第二行图开始，计算上一行图的2的幂次方
        # 获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            if j==1:
                #plt.ylabel('Decomposition Factor')
                #plt.xlabel('Time')
                plt.plot(map[re[j - 1]],color=(0,0,0.2,0.5))  # 列表从0开始
            else:
                plt.plot(map[re[j - 1]],color=(0,0,0.2,0.5))
    #plt.title('Alert')

    plt.show()

    return wp, map


def energy_sum(wp, n):
    # 小波包能量特征提取
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i, ord=None), 2))
    #return energy
    # print(re)
    # # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    # plt.figure(figsize=(10, 7), dpi=80)
    # # 再创建一个规格为 1 x 1 的子图
    # # plt.subplot(1, 1, 1)
    # # # 柱子总数
    # N = 8
    # values = energy
    # # # 包含每个柱子下标的序列
    # index = np.arange(N)
    # # # 柱子的宽度
    # width = 0.45
    # # # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    # p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    # # # 设置横轴标签
    # plt.xlabel('Node')
    # # 设置纵轴标签
    # plt.ylabel('Energy')
    # # # 添加标题
    # plt.title('WPE')
    # # # 添加纵横轴的刻度
    # plt.xticks(index, ('0', '1', '2', '3', '4', '5', '6', '7'))
    # #plt.yticks(np.arange(0, 10000, 10))
    # # # 添加图例
    # plt.legend(loc="upper right")
    # plt.show()
    return energy

# data_list：原始的数据列。 windows_size：窗口大小。  step：窗口步进长度
def window_cut(data_list, window_size, step):
    data_len = len(data_list)
    cut_num = int((data_len - window_size) / step)
    res = []
    for i in range(cut_num + 1):
        start_index = i * step
        end_index = start_index + window_size
        cur_data = data_list[start_index:end_index]
        # print(cur_data)
        res.append(cur_data)
        if i == cut_num:
            last_data = data_list[(start_index + step):]
            # print(last_data)
            res.append(last_data)
    return res


def normalize(data):
    # 调用 min 和 max 函数获取数据的最大值和最小值
    data_min = np.min(data)
    data_max = np.max(data)
    # 计算归一化后的数据
    data_normalized = (data - data_min) / (data_max - data_min)
    return data_normalized


def pre_process(path, i):
    '''数据预处理函数'''
    sample_data = pd.read_csv(path, header=0)
    data = np.array(sample_data)
    data_array = np.array(data[:, i])
    data = normalize(data_array).tolist()
    return data

# path='gaze/Fold1/12/gaze_semisleepy.txt'
# list_1=[]
# f = open(path, "r", encoding='utf-8')
# line = f.readline()
# #line=f.readline()
# while line:
#     list1 = line.split('\t')
#     list_1_data=eval(list1[1])
#     #list_1_data=eval(line)
#     #list_1_data=eval(line)
#     list_1.append(list_1_data)
#                     # list_2_data=eval(list1[1])
#                     # list_2.append(list_2_data)
#                     # list_3_data = eval(list1[3])
#                     # list_3.append(list_3_data)
#     line=f.readline()
# f.close()
# data=normalize(list_1).tolist()
# list_x = window_cut(data, 100, 60)
# res = []
# for j in range(len(list_x)):
#     wp_x, mp_x = wpd_plt(list_x[j], 1)
#     feature_x = energy_sum(wp_x, 1)
#     res.append(feature_x)
# rs = np.array(res)



if __name__ == '__main__':
    path1 = 'data'
    folds_list=os.listdir(path1)
    for f,fold in enumerate(folds_list):
        path2=path1+'/'+fold
        folder_list=os.listdir(path2)
        for ID,folder in enumerate(folder_list):
            path3=path2+'/'+folder
            fol_list=os.listdir(path3)
            for i,fols in enumerate(fol_list):
                path4=path3+'/'+fols
                list_1 = []
                #list_2 = []
                #list_3 = []
                f = open(path4, "r", encoding='utf-8')
                line = f.readline()
                while line:
                    #list1 = line.split('\t')
                    #list_1_data=eval(list1[0])
                    list_1_data=eval(line)
                    list_1.append(list_1_data)
                    # list_2_data=eval(list1[1])
                    # list_2.append(list_2_data)
                    # list_3_data = eval(list1[3])
                    # list_3.append(list_3_data)
                    line=f.readline()
                f.close()

                data=normalize(list_1).tolist()
                #data1=normalize(list_2).tolist()
                #data2 = normalize(list_3).tolist()
        #data_path = os.path.join(path1, path2)
        # 分帧滑窗
                list_x = window_cut(data, 50, 30)
                #list_y = window_cut(data1, 100, 60)
                #list_z = window_cut(data2, 100, 60)
            #list_y = window_cut(list_Y, 50, 20)
        # TimeFrequencyWP(list1, 250, wavelet='db4', maxlevel=8, s1='No.jpg', name='Normal sequence')
                res = []
                #re = []
                #r=[]
                for j in range(len(list_x)):
                    wp_x, mp_x = wpd_plt(list_x[j], 2)
                    feature_x = energy_sum(wp_x, 2)
                    # wp_y, mp_y = wpd_plt(list_y[j], 1)
                    # feature_y = energy_sum(wp_y, 1)
                    # wp_z, mp_z = wpd_plt(list_z[j], 1)
                    # feature_z = energy_sum(wp_z, 1)
                    #list_res = feature_x + feature_y
                    res.append(feature_x)
                    #re.append(feature_y)
                    #r.append(feature_z)
                #rs=np.concatenate((res,re),axis=1)
                rs=np.array(res)
                if (fols == 'ear-alert.txt'):
                    outputpath = 'xbearw50o30wei1/' + fold + '/'+folder +'/'+ 'xiaobo_1_w50_o30_alert.txt'

                elif (fols == 'ear-semisleepy.txt'):
                    outputpath = 'xbearw50o30wei1/' + fold + '/'+folder +'/'+ 'xiaobo_1_w50_o30_semisleepy.txt'
                elif (fols == 'ear-sleepy.txt'):
                    outputpath = 'xbearw50o30wei1/' + fold + '/'+folder +'/'+ 'xiaobo_1_w50_o30_sleepy.txt'
                np.savetxt(outputpath, rs, fmt="%.8f")

    print("OVER!")
