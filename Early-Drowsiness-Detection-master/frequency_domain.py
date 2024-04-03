import math
import os
import json
import cv2
import numpy as np
# import librosa.display
from python_speech_features import mfcc
from scipy import fftpack
# import scipy
import random
from matplotlib import pyplot as plt
import pickle
from utils.utils import create_directory

from sklearn.model_selection import train_test_split


class DataProcess(object):
    def __init__(self, gaze_locate_dir, save_dir, type='follow'):
        super(DataProcess, self).__init__()
        self.gaze_locate_dir = gaze_locate_dir
        self.save_dir = save_dir
        self.type = type
        # self.json_list = self.file_dir(self.data_dir, self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.fps = 25
        self.wlen = 10 * self.fps
        self.overlap = 2 * self.fps
        self.nfft = self.wlen
        self.follow_pair = {
            "11-14": [(3, 2)], "15-19": [(3, 2), (5, 4)], "20-25": [(3, 2)], "26": [(3, 2), (5, 4)],
            "27-29": [(3, 2)], "30-34": [(3, 2)], "35-40": [(5, 4)]
        }
        self.max_nf = 0
        self.gaze_flow_list = {}
        self.space_flow_list = {}
        self.p_data0, self.n_data0, self.p_data, self.n_data = [], [], [], []
        self.p_label, self.n_label = [], []
        self.data0, self.data, self.label = [], [], []
        self.train_test1 = {"train": [], "test": []}
        self.train_test2 = {"train": [], "test": []}
        self.train_test3 = {"train": [], "test": []}
        self.train_test4 = {"train": [], "test": []}
        self.train_test = [self.train_test1, self.train_test2, self.train_test3, self.train_test4]

    @staticmethod
    def load_json(json_dir):
        with open(json_dir, 'r') as fp:
            data = json.load(fp)
        fp.close()
        return data

    @staticmethod
    def obtain_fps(v_dir):
        cap = cv2.VideoCapture(v_dir)  # 读取原视频
        fps = cap.get(5)
        return int(fps)

    def search_label_pairs(self, name):
        if 11 <= int(name) <= 14:
            label_pairs = self.follow_pair["11-14"]
        elif 15 <= int(name) <= 19:
            label_pairs = self.follow_pair["15-19"]
        elif 20 <= int(name) <= 25:
            label_pairs = self.follow_pair["20-25"]
        elif 26 <= int(name) <= 26:
            label_pairs = self.follow_pair["26"]
        elif 27 <= int(name) <= 29:
            label_pairs = self.follow_pair["27-29"]
        elif 30 <= int(name) <= 34:
            label_pairs = self.follow_pair["30-34"]
        elif 35 <= int(name) <= 40:
            label_pairs = self.follow_pair["35-40"]
        return label_pairs

    @staticmethod
    def extract(loc_gazes):
        locates = {}
        frame_num = len(loc_gazes[list(loc_gazes.keys())[0]])
        for k in loc_gazes.keys():
            # print(len(loc_gazes[k]), frame_num)
            assert len(loc_gazes[k]) == frame_num
            locates[k] = [[0, 0] for j in range(frame_num)]
            loc_gazes[k] = [[np.array(loc_gazes[k][j][0]), np.array(loc_gazes[k][j][1])] for j in range(frame_num)]
            for i in range(frame_num):
                locates[k][i] = loc_gazes[k][i][0]
        return loc_gazes, locates

    @staticmethod
    def save_json(save_name, data):
        with open(save_name, 'w') as f:
            json.dump(data, f)
        f.close()

    def is_follower(self, r, s, k1, k, d=1000):
        n = len(r) // 25 + 2
        match = []
        for i in range(0, len(r) - 25, 25):
            r[i], s[i], r[i + 25], s[i + 25] = np.array(r[i]), np.array(s[i]), np.array(r[i + 25]), np.array(s[i + 25])
            dist = self.Dist(s[i], r[i])
            move_angle = np.dot((s[i] - r[i]), (r[i + 25] - r[i]))
            # move_direction = np.dot((r[i + 1] - r[i]), (s[i + 1] - s[i]))
            co_left = (s[i + 25][0] < s[i][0] < r[i][0]) and (s[i + 25][0] < r[i + 25][0] < r[i][0])
            co_right = (s[i + 20][0] > s[i][0] > r[i][0]) and (s[i + 25][0] > r[i + 25][0] > r[i][0])
            co_up = (s[i + 25][1] < s[i][1] < r[i][1]) and (s[i + 25][1] < r[i + 25][1] < r[i][1])
            co_down = (s[i + 25][1] > s[i][1] > r[i][1]) and (s[i + 25][1] > r[i + 25][1] > r[i][1])
            # if (k1, k) == (5, 4):
            #     print(r[i], s[i], r[i + 1], s[i + 1])
            #     # print(dist)
            #     print(move_angle)
            #     print(co_left, co_right, co_up, co_down)
            if dist < d and move_angle > 0 and (co_left or co_right or co_up or co_down):
                match.append(1)
            else:
                match.append(0)

        if (k1, k) == (3, 2) or (k1, k) == (5, 4):
            print("({}, {}): {}/{}".format(k1, k, np.sum(match), n))
        return 1 if np.sum(match) >= 0 * n else 0

    @staticmethod
    def Dist(p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def sigmoid_function(z):
        fz = []
        for num in z:
            fz.append(1 / (1 + math.exp(-num)))
        return fz

    @staticmethod
    def is_gaze(r1, s1):
        """
        :param r1: [[x1, y1], [g1, g2]]
        :param s1: [x1, y1]
        :return: 1 or 0
        """
        dist = r1[0] - s1
        if np.linalg.norm(r1[1]) * np.linalg.norm(dist) == 0:
            return 0
        angle = np.arccos(np.dot(r1[1], dist) / (np.linalg.norm(r1[1]) * np.linalg.norm(dist)))
        is_gaze = 1 if math.degrees(angle) <= 60 else 0
        # if angle <= 60:
        # score = 1 - 2 * math.degrees(angle) / 360
        # else:
        #     score = - angle / 10
        # print(math.degrees(angle))
        # print(angle)
        return is_gaze

    def Gaze_list(self, r, s):
        """
        :param r: [[[x1, y1], [g1, g2]], ...], trades length:n
        :param s: [[x1, y1], ...], trades length:n
        :return: G: whether r looks at s at each timestamp
        """
        n = len(r)
        G = []  # 一个片段的关注序列
        for i in range(n):
            # print(r[i], s[i])
            G.append(self.is_gaze(r[i], s[i]))
        return self.sigmoid_function(G)

    def Dist_list(self, r, s):
        # 一个片段内P与Q各时间点的距离
        # print(r[0], s[0])
        # last_angle = 0.0
        # D = []
        # for i in range(1, len(r)):
        #     x = np.array(s[i] - s[i-1])
        #     y = np.array(s[i] - r[i])
        #     Lx = np.linalg.norm(x)
        #     Ly = np.linalg.norm(y)
        #     if Lx * Ly > 0:
        #         cos_angle = x.dot(y) / (Lx * Ly)
        #     else:
        #         cos_angle = last_angle
        #     dist = Ly * cos_angle
        #     # print(Ly, cos_angle, dist)
        #     D.append(dist)
        #     last_angle = cos_angle
        D = [self.Dist(r[i], s[i]) for i in range(len(r))]
        return D

    def enframe1(self, loc_gazes0, locates0):
        loc_enframe, loc_gaze_enframe = {}, {}
        for k in sorted(locates0.keys()):
            frame_num = len(locates0[k])
            nf = (frame_num - self.wlen + self.overlap) // self.overlap  # 分成多少帧
            loc_gaze_enframe[k], loc_enframe[k] = [], []
            indf = np.multiply(self.overlap, np.array([i for i in range(nf)]))
            for i in range(nf):
                sub_loc_gaze = [[np.array(loc_gazes0[k][j][0]), np.array(loc_gazes0[k][j][1])] for j in
                                range(indf[i], indf[i] + self.wlen)]
                sub_loc = [np.array(locates0[k][j]) for j in range(indf[i], indf[i] + self.wlen)]
                loc_gaze_enframe[k].append(sub_loc_gaze)
                loc_enframe[k].append(sub_loc)
            # locates[k] = list(new_points)
        return loc_gaze_enframe, loc_enframe

    def enframe(self, init_gaze_list, init_dist_list, START_WINDOW):
        assert len(init_gaze_list) == len(init_dist_list)
        frame_num = len(init_gaze_list)
        gaze_list_enframe, dist_list_enframe = [], []
        overlap = self.overlap + START_WINDOW
        inc = self.wlen - self.overlap - 8
        nf = (2800 - self.wlen) // inc + 1  # 分成多少帧
        # nf = 15
        indf = np.multiply(inc, np.array([i for i in range(nf)]))
        for i in range(nf):
            sub_gaze = [init_gaze_list[j] for j in range(indf[i], indf[i] + self.wlen)]
            sub_loc = [init_dist_list[j] for j in range(indf[i], indf[i] + self.wlen)]
            gaze_list_enframe.append(sub_gaze)
            dist_list_enframe.append(sub_loc)
        # locates[k] = list(new_points)
        return gaze_list_enframe, dist_list_enframe

    def Gaze_Space_list(self, gaze_list_enframe, dist_list_enframe):
        n = len(gaze_list_enframe)
        gaze_feature_flow = np.zeros((n, self.wlen))
        space_feature_flow = np.zeros((n, self.wlen))
        for element, (gaze_list, dist_list) in enumerate(zip(gaze_list_enframe, dist_list_enframe)):
            # print(sub_len)
            # print(gaze_list)
            # self.plot_fft(gaze_list)
            # self.plot_fft(dist_list)
            gaze_feature_flow[element, :] = np.array(gaze_list)
            space_feature_flow[element, :] = np.array(dist_list)
        return gaze_feature_flow.reshape((n, self.wlen)), space_feature_flow.reshape((n, self.wlen))

    def MFCC(self, k1, k, feature_flow=None):
        y = np.absolute(np.fft.rfft(feature_flow, self.nfft))  # fft的幅度(magnitude)
        # print(feature_flow.shape)
        # print(y.shape)
        # self.plot_spectro(y)
        pow_frames = ((1.0 / self.nfft) * (y ** 2))  # Power Spectrum
        # print(pow_frames.shape)
        low_freq_mel = 0
        # 将频率转换为Mel，并取对数
        nfilt = 10
        high_freq_mel = (2595 * np.log10(1 + (self.fps / 2) / 700))
        # print(high_freq_mel)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

        bin = np.floor((self.nfft + 1) * hz_points / self.fps)

        # fbank特征
        num_ceps = 4
        fbank = np.zeros((nfilt, int(np.floor(self.nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # (n, 6)
        # self.plot_mfcc(mfcc, num_ceps)
        return mfcc

    def plot_fft(self, y):
        y = np.absolute(np.fft.fft(y, 512))
        plt.plot(y)
        plt.title('Spectrum')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_spectro(self, y):
        spec = librosa.amplitude_to_db(y, ref=np.max)
        librosa.display.specshow(spec, sr=self.fps, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

    def plot_mfcc(self, mfcc, num_ceps):
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
        librosa.display.specshow(mfcc, x_axis='time', sr=self.fps)
        plt.colorbar()
        plt.show()

    def train_test_save(self):
        assert len(self.p_data) == 180
        # random.shuffle(self.n_data)
        x_train1 = self.p_data[:125] + self.n_data[:125]
        x_train2 = self.p_data[55:] + self.n_data[55:180]
        x_train3 = self.p_data[:40] + self.p_data[95:] + self.n_data[:40] + self.n_data[95:180]
        x_train4 = self.p_data[:100] + self.p_data[155:] + self.n_data[:100] + self.n_data[155:180]
        ### gaze
        # x_train1 = self.p_data[20:80] + self.p_data[100:120] + self.p_data[135:] + self.n_data[20:80] + self.n_data[100:120] + self.n_data[135:180]
        # x_train2 = self.p_data[55:] + self.n_data[55:180]
        # x_train3 = self.p_data[:40] + self.p_data[95:] + self.n_data[:40] + self.n_data[95:180]
        # x_train4 = self.p_data[:40] + self.p_data[60:100] + self.p_data[120:165] + self.n_data[:40] + self.n_data[60:100] + self.n_data[120:165]
        ### space
        # x_train1 = self.p_data[20:40] + self.p_data[50:60] + self.p_data[70:100] + self.p_data[110:140] + self.p_data[145:] + self.n_data[20:40] + self.n_data[50:60] + self.n_data[70:100] + self.n_data[110:140] + self.n_data[145:180]
        # x_train2 = self.p_data[40:100] + self.p_data[110:175] + self.n_data[40:100] + self.n_data[110:175]
        # x_train3 = self.p_data[:30] + self.p_data[85:] + self.n_data[:30] + self.n_data[85:180]
        # x_train4 = self.p_data[:40] + self.p_data[60:100] + self.p_data[120:165] + self.n_data[:40] + self.n_data[60:100] + self.n_data[120:165]
        train_cross = [x_train1, x_train2, x_train3, x_train4]
        y_train = [1 for i in range(125)] + [0 for i in range(125)]

        x_test1 = self.p_data[125:] + self.n_data[125:180]
        x_test2 = self.p_data[:55] + self.n_data[:55]
        x_test3 = self.p_data[40:95] + self.n_data[40:95]
        x_test4 = self.p_data[100:155] + self.n_data[100:155]
        ### gaze
        # x_test1 = self.p_data[:20] + self.p_data[80:100] + self.p_data[120:135] + self.n_data[:20] + self.n_data[80:100] + self.n_data[120:135]
        # x_test2 = self.p_data[:55] + self.n_data[:55]
        # x_test3 = self.p_data[40:95] + self.n_data[40:95]
        # x_test4 = self.p_data[40:60] + self.p_data[100:120] + self.p_data[165:] + self.n_data[40:60] + self.n_data[100:120] + self.n_data[165:180]
        ### space
        # x_test1 = self.p_data[:20] + self.p_data[40:50] + self.p_data[60:70] + self.p_data[100:110] + self.p_data[140:145] + self.n_data[:20] + self.p_data[40:50] + self.p_data[60:70] + self.n_data[100:110] + self.n_data[140:145]
        # x_test2 = self.p_data[:40] + self.p_data[100:110] + self.p_data[175:] + self.n_data[:40] + self.n_data[100:110] + self.n_data[175:180]
        # x_test3 = self.p_data[30:85] + self.n_data[30:85]
        # x_test4 = self.p_data[40:60] + self.p_data[100:120] + self.p_data[165:] + self.n_data[40:60] + self.n_data[100:120] + self.n_data[165:180]
        test_cross = [x_test1, x_test2, x_test3, x_test4]
        y_test = [1 for i in range(55)] + [0 for i in range(55)]

        for i, (x_train, x_test) in enumerate(zip(train_cross, test_cross)):
            for (data0, label0) in zip(x_train, y_train):
                assert len(x_train) == len(y_train)
                tmp = []
                for data in list(data0):
                    tmp.append([float(d) for d in data])
                self.train_test[i]["train"].append((tmp, int(label0)))
            random.shuffle(self.train_test[i]["train"])

            for (data1, label1) in zip(x_test, y_test):
                assert len(x_test) == len(y_test)
                tmp = []
                for data in list(data1):
                    tmp.append([float(d) for d in data])
                self.train_test[i]["test"].append((tmp, int(label1)))
            name = self.save_dir + 'mfcc_gaze_10_4_w1021/'
            create_directory(name)
            self.save_json(name + str(i + 1) + '.json', self.train_test[i])

    def forward(self):
        yes, count, n = 0, 0, 0
        gaze_loc_list = sorted(os.listdir(self.gaze_locate_dir))
        # video_list = sorted(os.listdir(self.video_dir))
        # for (gaze_loc_json, video) in zip(gaze_loc_list, video_list):
        for gaze_loc_json in gaze_loc_list:
            count_tmp, yes_tmp = 0, 0
            name = gaze_loc_json.split('.')[0]
            gaze_loc_dir = os.path.join(self.gaze_locate_dir, gaze_loc_json)
            # v_dir = os.path.join(self.video_dir, video)
            # fps = self.obtain_fps(v_dir)

            print("Processing {}/{}".format(self.type, gaze_loc_json))
            loc_gazes0 = self.load_json(gaze_loc_dir)
            loc_gazes, locates = self.extract(loc_gazes0)
            label_pairs = self.search_label_pairs(name)

            # print(label_pairs)
            for k in sorted(locates.keys()):
                # 筛选id=k的人的位置跟随者
                for k1 in locates.keys():
                    if k == k1:
                        continue
                    # print((int(k1), int(k)))
                    positive = 0
                    if self.is_follower(locates[k1], locates[k], int(k1), int(k)) or (int(k1), int(k)) in label_pairs:
                        count_tmp += 1
                        count += 1
                        # print((int(k1), int(k)))
                        # 若k1是k的位置跟随者，判断是否为尾随行人对，并提取标签
                        if (int(k1), int(k)) in label_pairs:
                            label = 1
                            positive = 1
                            yes_tmp += 1
                            yes += 1
                            print("follow:{}->{}".format(int(k1), int(k)))
                        else:
                            label = 0
                        # 初始注视序列, 距离序列（类比原始语音信号）
                        # 注视序列，需要k1的位置与注视方向，k的位置
                        init_gaze_list = self.Gaze_list(loc_gazes[k1], locates[k])
                        # print(init_gaze_list)
                        # print(len(init_gaze_list))
                        # 距离序列，需要k1与k的位置
                        init_dist_list = self.Dist_list(locates[k1], locates[k])

                        # 自写分步计算
                        for START_WINDOW in (0, 2, 4, 6, 8):
                            # 分帧, 重复5次
                            gaze_list, dist_list = self.enframe(init_gaze_list, init_dist_list, START_WINDOW)
                            # if (int(k1), int(k)) in [(3, 2), (5, 4)]:
                            #     self.plot_fft(gaze_list[0])
                            # self.plot_fft(init_dist_list)
                            # 注视、距离子序列矩阵
                            gaze_feature_flow, space_feature_flow = self.Gaze_Space_list(gaze_list, dist_list)
                            mfcc_gaze_feature = self.MFCC(k1, k, gaze_feature_flow)
                            mfcc_feature = mfcc_gaze_feature
                            # mfcc_space_feature = self.MFCC(k1, k, space_feature_flow)
                            # mfcc_feature = mfcc_space_feature
                            # if aggregate
                            # mfcc_feature = np.concatenate((mfcc_gaze_feature, mfcc_space_feature), axis=1)
                            # print(mfcc.shape)

                            # 2. 直接调用函数计算
                            # mfcc_feature = mfcc(np.array(init_dist_list), self.fps, winlen=5, winstep=2, nfilt=8, nfft=self.wlen)
                            # print(mfcc_feature.shape)
                            if label == 1:
                                self.p_data.append(mfcc_feature)
                            else:
                                self.n_data.append(mfcc_feature)
            print("{}: locate follow pairs/hidden pairs: {}/{}".format(gaze_loc_json, count_tmp, yes_tmp))

        print("total loc pairs/hidden pairs: {}/{}".format(count, yes))
        print(len(self.p_data), len(self.n_data))
        self.train_test_save()


if __name__ == '__main__':
    gaze_locate_folder = '/tsc/TWIESN/follow1/'
    save_folder = '/tsc/TWIESN/data/'
    DP = DataProcess(gaze_locate_folder, save_folder, type='follow')
    DP.forward()
