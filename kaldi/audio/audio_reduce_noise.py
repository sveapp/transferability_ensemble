# !/usr/bin/env python
import os

import numpy as np
import wave
import math

from wsj_recipe.steps.libs.common import logger


# def optimize_audio(input_file, output_file):
#     """
#     对音频进行降噪处理，隔离可听见的声音。将低通滤波器与高通滤波器结合使用。
#     过滤掉200hz及以下的内容，然后过滤掉3000hz及以上的内容，可以很好地保持可用的语音音频。
#     :param input_file: 原始文件
#     :param output_file: 处理后文件
#     :return:
#     """
#     if not os.path.exists(input_file):
#         logger.error("文件不存在，请检查文件: %s" % input_file)
#
#     if os.path.isfile(output_file) and os.path.exists(output_file):
#         os.remove(output_file)
#
#     cmd = 'ffmpeg -i %s -af "highpass=f=200, lowpass=f=3000" %s' % (input_file, output_file)
#     subprocess_cmd(cmd, "handle_audio")
#
#     return output_file

def nextpow2(n):
    """
    求最接近数据长度的2的整数次方
    An integer equal to 2 that is closest to the length of the dat
    Eg:
    nextpow2(2) = 1
    nextpow2(2**10+1) = 11
    nextpow2(2**20+1) = 21
    """
    return np.ceil(np.log2(np.abs(n))).astype('long')


# 打开WAV文档
f = wave.open("../csv_to_wav/adv/adversarial_yuxuan(ok).wav")
params = f.getparams()  # 读取格式信息 (nchannels, sampwidth, framerate, nframes, comptype, compname)
nchannels, sampwidth, framerate, nframes = params[:4]
fs = framerate
str_data = f.readframes(nframes)  # 读取波形数据
f.close()

x = np.frombuffer(str_data, dtype=np.short)  # 将波形数据转换为数组

len_ = 20 * fs // 1000  # 样本中帧的大小 #160
PERC = 50  # 窗口重叠占帧的百分比
len1 = len_ * PERC // 100  # 重叠窗口 #80
len2 = len_ - len1  # 非重叠窗口 #80

# 设置默认参数
Thres = 3
Expnt = 2.0
beta = 0.06
G = 0.9
# 初始化汉明窗
win = np.hamming(len_)
# normalization gain for overlap+add with 50% overlap
winGain = len2 / sum(win)  # winGain = {float64} 0.9308820107051431

# Noise magnitude calculations - assuming that the first 5 frames is noise/silence
nFFT = 2 * 2 ** (nextpow2(len_))  # 512
noise_mean = np.zeros(nFFT)  # np(512)

j = 0
for k in range(1, 6):
    noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
    j = j + len_
noise_mu = noise_mean / 5

# --- allocate memory and initialize various variables
k = 1
img = 1j
x_old = np.zeros(len1)  # np(80)
Nframes = len(x) // len2 - 1  # 405
xfinal = np.zeros(Nframes * len2)


def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 10:
            a = 1
    return a


def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 10:
            a = 1
    return a


def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list


# =========================    Start Processing   ===============================
for n in range(0, Nframes):
    # Windowing
    insign = win * x[k - 1:k + len_ - 1]  # np(160)
    # compute fourier transform of a frame
    spec = np.fft.fft(insign, nFFT)  # np(512)
    # compute the magnitude
    sig = abs(spec)

    # save the noisy phase information
    theta = np.angle(spec)
    SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

    if Expnt == 1.0:  # 幅度谱
        alpha = berouti1(SNRseg)
    else:  # 功率谱
        alpha = berouti(SNRseg)
    # 根据公式求出每一帧的去噪后的幅值sub_speech。
    sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
    # 当纯净信号小于噪声信号的功率时
    diffw = sub_speech - beta * noise_mu ** Expnt

    # beta negative components

    z = find_index(diffw)
    if len(z) > 0:
        # 用估计出来的噪声信号表示下限值
        sub_speech[z] = beta * noise_mu[z] ** Expnt
        # --- implement a simple VAD detector --------------
    if SNRseg < Thres:  # Update noise spectrum
        noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
        noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
    # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
    # 交换上下对称元素
    sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
    x_phase = (sub_speech ** (1 / Expnt)) * (
            np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
    # take the IFFT

    xi = np.fft.ifft(x_phase).real
    # --- Overlap and add ---------------
    xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
    x_old = xi[0 + len1:len_]
    k = k + len2
# 保存文件
wf = wave.open('en_outfile.wav', 'wb')
# 设置参数
wf.setparams(params)
# 设置波形文件 .tostring()将array转换为data
wave_data = (winGain * xfinal).astype(np.short)
wf.writeframes(wave_data.tobytes())
wf.close()
