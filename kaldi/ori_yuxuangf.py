# coding=utf-8
import argparse
import librosa
import math
import os
import time
import wave
from math import sqrt
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.stats as st
from scipy.signal import butter, lfilter

from audio.audio_test import clip, roll, speed_librosa, pitch_shifting
from audio.audio_test import log10, energy_tensor
from audio.audio_test import np_SNR, np_PSNR
from audio.audio_test import wavwrite_for_noise, db_chage
from audio.base_model import acoustic_model
from audio.base_model import l0reformat
from audio.base_model import tdnn
from transferability_check.aliyun import aliyun
# from transferability_check.api import SpeechBrain
from transferability_check.api import deepspeech_api
from transferability_check.baidu import baidu
from transferability_check.xfyun import xfyun
from transferability_check.tencent import tencentyun

# import google.cloud.speech_v1p1beta1 as speech
# from base_model import google_api_phone
# from numba import jit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置为1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出

import tensorflow as tf

# GPU设置
print(tf.test.is_gpu_available())
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存

CONFIDENCE = 0.0  # 10,30,50,60,80 # how strong the adversarial example should be EPS
EPS = 1.0
K = 1.0
momentum = 1.0
boxmin = 0.
boxmax = 1.
epsilon = 1e-08
boxmul = (boxmax - boxmin) / 2.  # 0.5
boxplus = (boxmin + boxmax) / 2.  # 0.5

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--value_noise', type=int, default=4000, help="Input scale noise ")
parser.add_argument('--sigma', type=int, default=4000, help="Input value of normal's noise ")
parser.add_argument('--noise_class', type=str, default='uniform', help="class")
parser.add_argument('--iteration_time', required=False, type=int, default=1, help="iteration_time")
parser.add_argument('--music_path', type=str,
                    default='/data/guofeng/Ensembleattck/audio_carry/4s/01_English_0070.wav', help="music_path")
parser.add_argument('--music_num', required=False, type=int, default=1, help="music_num ")
parser.add_argument('--num_m', type=int, default=5, help="m")
parser.add_argument('--prob', default=0.5, help="probability for input diversity ")
parser.add_argument('--dropout', default=0.5, help="the nums of close sig ")
parser.add_argument('--momentum', default=1.0, help="momentum about the audio_model. ")
parser.add_argument('--sr1', default=0.95, help="momentum about the audio_model. ")
parser.add_argument('--sr2', default=0.25, help="momentum about the audio_model. ")
parser.add_argument('--ivector_path', default='/data/guofeng/Ensembleattck/kaldi/mfcc/musicivectors_ori.csv',
                    help="ivector")
parser.add_argument('--target_path',
                    default='/data/guofeng/Ensembleattck/kaldi/fgm_data/pdf_aspire533680242_okay_google_turn_off_the_light.csv',
                    help="target")
args = parser.parse_args()


def q_mean(num):
    return sqrt(sum(n * n for n in num) / len(num))


def wav_write(iteration_ref, loss_ref, iteration_time, music_num, wav_data):
    RATE = 16000

    # GENERATE MONO FILE #
    wavpath = "csv_to_wav/yuxuan24/music%s_sample_iteration%s_loss%s_iterationnumber%s_time.wav" % (
        music_num, iteration_ref, loss_ref, iteration_time)
    wv = wave.open(wavpath, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = np.array(wav_data)
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wavpath


def wav_write_for_libri_speech():
    RATE = 16000
    # GENERATE MONO FILE #
    wavpath = "csv_to_wav/adversarial_yuxuangf.wav"
    wv = wave.open(wavpath, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = np.loadtxt(open("zy_0102/mfcc0102/yuxuangf_finish.csv", "r"), delimiter=",", dtype="float32")
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wavpath


def wav_write_for_ae(wav_write_path, new_wav):
    RATE = 16000
    # GENERATE MONO FILE #
    wv = wave.open(wav_write_path, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = new_wav
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wav_write_path


def wav_write_for_noise(wav_write_path, new_wav):
    RATE = 16000
    # GENERATE MONO FILE #
    wv = wave.open(wav_write_path, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = new_wav
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wav_write_path


def frame_mask(VP, num_segments_t, hop_size, seg_size):
    if VP is True:
        sr1 = int(args.sr1 * num_segments_t)
        sr2 = int(args.sr2 * num_segments_t)
        sr_one = np.random.binomial(1, 1, sr1)
        sr_zero = np.random.binomial(1, 0, sr2)
        nk = int(1 / (args.sr1 + args.sr2)) + 1
        if nk < 6:
            mask_f = np.append(sr_one, sr_zero)
            mask_f = np.append(mask_f, sr_one)
            mask_f = np.append(mask_f, sr_zero)
            mask_f = np.append(mask_f, sr_zero)
            mask_f = np.append(mask_f, sr_zero)
            mask_f = np.append(mask_f, np.random.binomial(1, 0, np.abs(num_segments_t - mask_f.shape[0])))
        mask_f = mask_f[:num_segments_t]
    else:
        mask_f = np.random.binomial(1, args.sr1, size=num_segments_t)

    # print(mask_f.shape, num_segments_t)
    wav_like = np.ones_like(original)
    for y in range(0, num_segments_t):
        idx_s = np.int32(hop_size * y)
        idx_e = np.int32(seg_size + hop_size * y)
        if mask_f[y] == 1:
            wav_like[idx_s:idx_e] = 1
        else:
            wav_like[idx_s:idx_e] = 0
    mask = wav_like
    return mask
    # print(mask, mask.shape)


def add_noise(shape_in, best_sig=None, mu=0, RO=False):
    global args
    noise_class = args.noise_class
    sigma = args.sigma
    value_noise = args.value_noise
    print("noise:", noise_class, sigma, value_noise)
    if noise_class == 'normal':
        if (RO is True) and (best_sig is not None):
            put_noise = tf.random.normal(shape=shape_in, mean=best_sig, stddev=sigma)
        else:
            put_noise = tf.random.normal(shape=shape_in, mean=mu, stddev=sigma)
    elif noise_class == 'uniform':
        put_noise = tf.random_uniform(shape=shape_in, minval=-value_noise, maxval=value_noise)
    else:
        raise ValueError("only normal or uniform can be selected")
    return put_noise


def up_down_noise_db(x_wav, db, NR):
    """
    just we using the function ,when NR is true ,means noise up,db >0,else NR is false,means db<0
    @param x_wav:
    @param db:
    @param NR:
    @return:
    """
    if NR:
        delta_value_fixed = x_wav - original
        temp_wav = 'ori_audio/temp1.wav'
        temp_wav_path = wavwrite_for_noise(temp_wav, delta_value_fixed)  # noise wav
        db = db  # reduce or up db 20
        out_temp = 'ori_audio/temp11.wav'
        pa1 = db_chage(temp_wav_path, db, out_temp)  # reduce noise wav db
        (rate_temp, delta_value_fixed) = wav.read(pa1)
        x = original + delta_value_fixed
    if not NR:
        delta_noise = x_wav - original
        delta_value_fixed = original
        temp_wav = 'ori_audio/temp2.wav'
        temp_wav_path = wavwrite_for_noise(temp_wav, delta_value_fixed)  # self.self.self.original wav
        db = db  # reduce or up db 20
        out_temp = 'ori_audio/temp22.wav'
        pa1 = db_chage(temp_wav_path, db, out_temp)  # reduce self.self.self.original wav db
        (rate_temp, delta_value_fixed) = wav.read(pa1)
        x = delta_noise + delta_value_fixed
    return x


def norm(mfcc_in):
    shape = mfcc_in.shape[1]  # 40
    mfcc_mean, mfcc_std = tf.nn.moments(x=mfcc_in, axes=1)
    mfcc_mean = tf.reshape(mfcc_mean, shape=(mfcc_mean.shape[0], 1))
    mfcc_std = tf.reshape(mfcc_std, shape=(mfcc_std.shape[0], 1))
    mfcc_mean_tile = tf.tile(mfcc_mean, [1, shape])
    mfcc_std_tile = tf.tile(mfcc_std, [1, shape])
    mfcc_out = (mfcc_in - mfcc_mean_tile) / (mfcc_std_tile + 1e-14)
    return mfcc_out


def tf_input_diversity(input_tensor):
    global args
    input_length = int(input_tensor.shape[0])
    if input_length < 36000:
        shift_l = tf.random_uniform(shape=[1], minval=2, maxval=128, dtype=tf.int32)[0]
        padded = tf.pad(input_tensor, [[shift_l, 0]], constant_values=0.)
        padded = tf.pad(padded, [[0, 36000 - input_length - shift_l]], constant_values=0.)
        padded.set_shape(36000, )
        # print("paded:", padded)

        input_tensor = tf.pad(input_tensor, [[0, 36000 - input_length]], constant_values=0.)
        ret = tf.cond(tf.random.uniform(shape=[1])[0] < args.prob, lambda: padded, lambda: input_tensor)
        ret.set_shape(36000, )
        # print("ret:", ret, ret.shape)
    else:
        raise ValueError("max length can not beyond 36000 ")
    return ret


def np_input_diversity(input_np):
    global args
    # rescaled = speed_numpy(input_np)
    rescaled = speed_librosa(input_np)

    rescaled = pitch_shifting(input_np)
    rescaled = roll(rescaled)

    shift_l = np.random.randint(low=2, high=32, size=[1], dtype=np.int32)[0]
    padded = np.pad(rescaled, (shift_l, 0), constant_values=0.0)

    # shift_l2 = np.random.randint(low=2, high=128, size=[1], dtype=np.int32)[0]
    # padded = np.pad(padded, (0, shift_l2), constant_values=0.)

    if np.random.uniform(size=[1])[0] < args.prob:
        ret = clip(padded, len(input_np))
    else:
        ret = input_np

    return ret


def dropout(x_numpy):
    global args
    drop_prob = args.dropout
    keep_prob = 1.0 - drop_prob
    x_numpy = x_numpy * np.random.binomial(1, keep_prob, x_numpy.shape)
    # print("dropout:", x_numpy.shape)
    return x_numpy


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen).astype(np.float32)  # x = {ndarray: (15,)} .
    kern1d = st.norm.pdf(x)  # {ndarray: (15,)}
    kernel = kern1d / kern1d.sum()
    stack_kernel = np.reshape(kernel, (kernlen, 1, 1))
    stack_kernel = stack_kernel.astype(np.float32)
    return stack_kernel


def k_smooth(x_sig, k=100):
    x_len = x_sig.shape[0]
    x_sig = tf.cast(x_sig, tf.float32)
    kernel = np.full(shape=(k, 1, 1), fill_value=1 / k, dtype=np.float32)
    x = tf.reshape(x_sig, (1, x_len, 1))
    conv1d = tf.nn.conv1d(x, kernel, stride=1, padding='SAME')
    x = tf.reshape(conv1d, (x_len,))

    return x


def top_k(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
    return topk_data, topk_index


def pl(loss):
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss, label="$loss$")
    plt.legend()
    plt.show()


def pl_snr(snr_list):
    plt.title("snr")
    plt.xlabel("epoch")
    plt.ylabel("snr")
    plt.plot(snr_list, label="$snr$")
    plt.legend()
    plt.show()


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max() + epsilon)
    elite_pop_probs = scores_logits / scores_logits.sum()
    # print("elite_pop_scores", elite_pop_scores)
    # print("scores_logits", scores_logits)
    # print("elite_pop_probs", elite_pop_probs)
    cand1 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5
    next_pop = mask * cand1 + (1 - mask) * cand2
    return next_pop


def mutate_pop(pop, mutation_p, noise_stdev, elite_pop):
    noise = np.random.randn(*pop.shape) * noise_stdev
    noise = highpass_filter(noise)
    mask = np.random.rand(pop.shape[0], elite_pop.shape[1]) < mutation_p
    new_pop = pop + noise * mask
    return new_pop


def featrue_prob(x, num_segments_t, ivectors, TF_NOISE):
    noise_in = add_noise(shape_in=x.shape, mu=0, best_sig=None, RO=False)
    sig2 = tf.cond(TF_NOISE, lambda: x + noise_in, lambda: x)  # if tf.cond(TF_NOISE):  # 不要使用‘is true’这样的语句
    # =============================================================================
    # sig2 = input_diversity(sig2)  # tensorflow input diversity
    mfcc, _ = acoustic_model(sig2, num_segments_t)  # input_diversity for sig2
    # mfcc = acoustic_model(sig2, num_segments_t)  # mfcc = {Tensor} Tensor("mul_1700:0", shape=(40, 425), dtype=float32)
    frame_length = tf.shape(mfcc)[1]
    frame_length = tf.cast(frame_length, dtype=tf.float32)
    ses1 = tf.Session()
    frame_length_out = np.int32(ses1.run(frame_length))
    print("#### ", frame_length_out)
    ses1.close()
    t_zeros = tf.zeros(shape=[40, 500 - frame_length_out], dtype=tf.float32)
    mfcc = tf.concat([mfcc, t_zeros], axis=1)
    mfcc = tf.transpose(mfcc)
    # mfcc = norm(mfcc)
    # =============================================================================
    dnnInput = l0reformat(mfcc, ivectors)
    oriOut = tdnn(dnnInput, 50)
    print("mfcc shape:", mfcc.shape)
    return mfcc, oriOut, frame_length_out


def ods_loss(ori_prob, target):
    preds = ori_prob[0, :, :]

    out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)
    dnn_max = tf.reduce_max(out_dnn, axis=1, keepdims=True)

    targets_index = target
    lenl = len(targets_index)
    loss = tf.constant(0, dtype=tf.float32)

    for i in range(0, lenl):
        ran_d = tf.random.uniform(shape=(8629,), minval=-1, maxval=1, dtype=tf.float32)
        loss_delta = tf.multiply(out_dnn[i, :], ran_d)
        loss = tf.add(loss, loss_delta)
    loss = -loss

    return loss


def label_smooth_cross_loss(ori_prob, target):
    preds = ori_prob[0, :, :]
    preds = tf.nn.softmax(preds)
    out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)

    targets_index = target
    lenl = len(targets_index)
    loss = tf.constant(0, dtype=tf.float32)

    for i in range(0, lenl):
        label = tf.Variable(tf.zeros(shape=[1, out_dnn.shape[1]]))
        label = tf.scatter_nd_update(label, indices=[[0, targets_index[i]]], updates=[1])
        label = tf.reshape(label, (8629,))
        loss_delta = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=out_dnn[i, :],
                                                     label_smoothing=0.25)
        loss = tf.add(loss, loss_delta)
    loss = -loss

    return loss


def cross_loss(ori_prob, target):
    preds = ori_prob[0, :, :]
    preds = tf.nn.softmax(preds)
    out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)
    targets_index = target

    lenl = len(targets_index)
    loss = tf.constant(0, dtype=tf.float32)

    for i in range(0, lenl):
        label = tf.Variable(tf.zeros(shape=[1, out_dnn.shape[1]]))
        label = tf.scatter_nd_update(label, indices=[[0, targets_index[i]]], updates=[1])
        label = tf.reshape(label, (8629,))
        loss_delta = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out_dnn[i, :])
        loss = tf.add(loss, loss_delta)
    loss = -loss

    return loss


def tanh_space(x):
    a = tf.reduce_min(tf.convert_to_tensor(x))
    b = tf.reduce_max(tf.convert_to_tensor(x))
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    # 先使用极差变化法等比列缩放到0,1区间
    xx = (x - a) / (b - a + 1e-20)  # To prevent denominatorx is 0
    # 再使用tanh方法将其归一化
    x_new = tf.tanh(xx) * boxmul + boxplus

    return x_new


def soft_np_clip(delta, m=1.0):
    delta_stft = librosa.stft(delta)
    # delta_eq = (delta_stft / (np.abs(delta_stft) + epsilon)) * m
    # delta_eq = np.clip(delta_stft,-(np.min(delta_stft)+np.max(delta_stft))/2,(np.min(delta_stft)+np.max(delta_stft))/2)
    delta_eq = delta_stft + np.random.uniform(0.0, 1.0, delta_stft.shape)
    delta_apply = librosa.griffinlim(delta_eq)
    return delta_apply


def saliency(x, ori_prob, target, clip_max, clip_min):  # 可以将此算法放入遗传算法或是其他算法中配合使用
    global args

    ind_K = math.ceil((K / 8629) * int(x.shape[0]))
    eps = EPS

    preds = ori_prob[0, :, :]
    targets_index = target
    lenl = len(targets_index)

    out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)

    mat = []
    for a in range(0, lenl):
        # print(out_dnn, a, type(a), targets_index[a], type(targets_index[a]))
        lab_target = tf.gather_nd(out_dnn, [a, targets_index[a]])
        # different tensorflow version,some need (a, targets_index[a]) type of tensor ,else type of python or tensor
        # lab_target = out_dnn[a, targets_index[a]]
        mat.append(lab_target)
    mat_target = tf.convert_to_tensor(mat)
    print(mat_target)
    mat_target = tf.expand_dims(mat_target, 1)
    true_max = tf.math.top_k(out_dnn, k=1, sorted=True)[0]
    true_max = true_max[0:lenl]
    dy_dx, = tf.gradients(true_max, x)
    dt_dx, = tf.gradients(mat_target, x)

    print(" dt_dx  dy_dx", dt_dx, dy_dx,)
    # x_c = (x - clip_min + epsilon) / (clip_max - clip_min + epsilon)
    x_c = tf.divide((x - clip_min + epsilon), (clip_max - clip_min + epsilon))
    ct = tf.logical_or(eps < 0.0, x_c <= 1.0)
    cd = tf.logical_or(eps > 0.0, x_c >= 0.0)
    mask = tf.reduce_all([dt_dx >= 0.0, dy_dx <= 0.0, ct, cd], axis=0)
    mask = tf.cast(mask, tf.float32)

    sources = mask * dt_dx * tf.abs(dy_dx)

    ind = tf.math.top_k(sources, k=ind_K, sorted=True)[1]
    ind_mask = tf.zeros_like(x)
    for a in range(ind_K):
        ind_mask += tf.one_hot(ind[a], int(x.shape[0]), on_value=1.0, off_value=0.0)
    print("finish saliency")
    # dx = tf.random_uniform(shape=len(x), minval=-args.value_noise, maxval=args.value_noise)
    # advx = x + ind_mask * dx

    return mask, ind_mask


def API():
    fp_loss = open('mfcc/mfcc1219/yuxuangf_loss.txt', 'w')
    baidu_reg_result = open('transferability_check/baidu/baidu.txt', 'w')
    # xfyun_reg_result = open('transferability_check/xfyun/xfyun.txt', 'w')
    # aliyun_reg_result = open('transferability_check/aliyun/aliyun.txt', 'w')
    xfyun_reg_result = None
    aliyun_reg_result = None
    tencent_reg_result = open('transferability_check/tencent/tencent.txt', 'w')
    deepspeech_reg_result = None
    speechbrain_reg_result = open('transferability_check/api/speechbrain.txt', 'w')
    # speechbrain_reg_result = None
    # deepspeech_reg_result = open('transferability_check/api/deepspeech.txt', 'w')
    return fp_loss, baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result, deepspeech_reg_result, speechbrain_reg_result


def cer(ref, hyp):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation

    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref)][len(hyp)] / float(len(ref))


def get_cer(path, sentence):
    CER_list = []
    target = sentence.split(' ')
    with open(path, mode='r') as fp:
        text_list = fp.readlines()
        for text in text_list:
            text = text.strip('\n').strip(' ').split(' ')
            CER_list.append(cer(target, text))
    return CER_list, np.mean(CER_list)


class Attack():
    def __init__(self, sig_in, sess, mfcc_shape):
        """
        @type sess: object
        """
        # super(Attack, self).__init__(mfcc_shape)
        self.sig_in = sig_in
        self.original = sig_in
        self.MAX = np.max(sig_in)
        self.MIN = np.min(sig_in)
        self.sess = sess
        self.h0_frame = 28
        self.n_feature = 220
        self.n_pdfid = 8629
        self.n = 50
        self.FS = 16000
        self.seg_size = self.FS * 0.025
        self.hop_size = self.FS * 0.01
        self.num_segments_t = math.floor((len(self.sig_in) - self.seg_size) / self.hop_size) + 1
        self.length = len(self.sig_in)
        self.targets = np.loadtxt(
            open('/data/guofeng/Ensembleattck/kaldi/fgm_data/pdf_aspire533680242_okay_google_turn_off_the_light.csv',
                 "rb"),
            delimiter=" ", dtype="int16")
        # self.IvectorInput = np.loadtxt(
        #     open("/data/guofeng/Speech_Recognition/mfcc/mfcc1219/musicivectors_ori.csv", "rb"), delimiter=" ",
        #     dtype="float32")

        # self.IvectorInput = np.loadtxt(open("mfcc/musicivectors_ori.csv", "rb"), delimiter=" ", dtype="float32")
        self.input_shape = self.length

        self.sig = tf.placeholder(tf.float32, shape=self.length)
        # self.ivectors = tf.placeholder(tf.float32, shape=(None, 100))
        self.ivectors = np.loadtxt(
            open("/data/guofeng/Ensembleattck/kaldi/mfcc/musicivectors_ori.csv", "rb"),
            delimiter=" ", dtype="float32")
        self.adam_grad_tf = tf.placeholder(tf.float32, shape=self.input_shape)
        # self.delta_apply = tf.placeholder(tf.float32, shape=self.input_shape)
        self.NOISE = tf.placeholder(tf.bool)
        zero = tf.zeros(self.input_shape, dtype=tf.float32)
        self.m = tf.get_variable(name='m', dtype=tf.float32, initializer=zero)
        self.v = tf.get_variable(name='v', dtype=tf.float32, initializer=zero)
        self.t = tf.Variable(dtype=tf.float32, initial_value=1.0)
        self.delta = tf.get_variable(name='delta', dtype=tf.float32, initializer=zero)
        # self.gradOld_tf = tf.get_variable(name='gradOld_tf',dtype=tf.float32,initializer=zero)
        self.gradOld_tf = np.zeros(dtype=np.float32, shape=self.input_shape)
        # self.gradOld_tf = np.zeros(shape=self.input_shape, dtype=np.float32)
        # self.m_adam = tf.placeholder(tf.float32, shape=self.input_shape)
        # self.v_adam = tf.placeholder(tf.float32, shape=self.input_shape)
        # self.t_adam = tf.placeholder(tf.float32)
        # self.gradOld_tf = tf.placeholder(tf.float32, shape=self.input_shape)
        # self.grad_old_np = np.zeros(shape=self.input_shape, dtype=np.float32)
        # self.m = np.zeros(self.input_shape, dtype=np.float32)
        # self.v = np.zeros(self.input_shape, dtype=np.float32)
        # self.t = 1.0
        # self.delta = 0.0
        # self.delta = tf.convert_to_tensor(self.delta)
        # self.saliency_mask = tf.ones_like(self.sig, dtype=tf.float32)
        # self.temp_ind_mask = tf.ones_like(self.sig, dtype=tf.float32)
        self.l, self.g, self.l1, self.l1_dist, self.l2_dist, self.vv, self.dnn_max = self.Get_grad(self.sig,
                                                                                                   self.ivectors,
                                                                                                   self.NOISE)
        _, self.apply_delta = self.adamOpt(x=self.sig, grad=self.adam_grad_tf)
        self.adv = self.original + self.apply_delta

    def adamOpt(self, x, grad, lr=100, beta1=0.9, beta2=0.999, epsilon=1e-08, clip_min=-32767, clip_max=32767):
        global original
        m = self.m
        v = self.v
        t = self.t + 1
        # t = self.t
        grads = grad

        # global STEP
        # STEP += 1
        # if STEP <= warm_up:
        #     lr_t = lr * (STEP / warm_up)
        # else:
        #     lr_t = tf.train.exponential_decay(lr, t - warm_up, 200, 0.95)

        lr_t = lr * tf.sqrt(1 - tf.pow(beta2, t)) / (1 - tf.pow(beta1, t))
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * tf.square(grads)
        pertuabations = tf.math.ceil(lr_t * m / (tf.sqrt(v) + epsilon))  # 向上取整
        pertuabations_fixed = (momentum * self.delta + pertuabations)
        pertuabations_fixed = tf.clip_by_value(pertuabations_fixed, -5000, 5000)
        x = x + pertuabations_fixed
        # s1 = tf.Session()
        # delta_value = s1.run(pertuabations_fixed)
        # s1.close()
        # delta_apply = soft_np_clip(delta_value)
        # x = x +  delta_apply
        # means = tf.reduce_sum(delta_value) / tf.cast(tf.count_nonzero(delta_value), tf.float32)
        # mxa = tf.reduce_max(delta_value)
        # mni = tf.reduce_min(delta_value)
        # delta_value_1 = tf.where(delta_value > means, x=tf.clip_by_value(delta_value, 0.5 * means, means * 1.5),
        #                          y=delta_value)
        # delta_value = x - self.original
        # s1 = tf.Session()
        # delta_value = s1.run(delta_value )
        # s1.close()
        # delta_apply = soft_np_clip(delta_value)

        # x = self.original + delta_apply
        #  using tanh function to clip x range from min to max 'x = clip_log(x, self.MIN, self.MAX)'
        if (clip_min is not None) and (clip_max is not None):
            x = tf.clip_by_value(x, clip_min, clip_max)

        #  you must understand tf.assign and '=' assignment statement you must use tf.assign to change a variable 'self.delta = pertuabations_fixed'
        self.delta = tf.assign(self.delta, pertuabations_fixed)
        self.m = tf.assign(self.m, m)
        self.v = tf.assign(self.v, v)
        self.t = tf.assign(self.t, t)
        apply_delta = x - self.original
        return x, apply_delta

    def dbdist_loss(self, x):  # bug is input 'tanh_space' parameter x can't 0
        global original
        put_noise = tf.random.normal(shape=(len(self.original),), mean=0, stddev=0.1)  # using noise to enhance x
        delta_v = tanh_space(x + put_noise) - tanh_space(self.original)
        # delta_v = (x + put_noise) - self.original
        dbdist_loss = 20.0 * log10(energy_tensor(delta_v, len(self.original)))
        tt = self.delta
        return dbdist_loss, tt

    def l2dist_loss(self, x):
        global original
        # l2dist = tf.reduce_sum(x - self.original)
        # l2dist = tf.reduce_sum(tf.square(tanh_space(x) - tanh_space(self.original)))
        l2dist = tf.reduce_sum(tf.square(x - self.original)) * 1 / x.shape.as_list[0]
        return l2dist

    def l2_dist_cal(self, x):
        global original
        # l2dist = tf.reduce_sum(x - self.original)
        # l2dist = tf.reduce_sum(tf.square(tanh_space(x) - tanh_space(self.original)))
        # print("!!!!!2",x.shape[0], x.shape, type(x.shape[0], type(x.shape), type(x)))
        l2dist = tf.sqrt(tf.reduce_sum(tf.square(x))) / x.shape.as_list()[0]
        # l2dist = np.linalg.norm(x) / x.shape.as_list()[0]

        return l2dist

    def l1_dist_cal(self, x):
        global original
        # print("!!!!!1", x.shape[0], x.shape, type(x.shape[0]), type(x.shape), type(x))
        l1_loss = tf.reduce_sum(tf.abs(x - self.original)) / x.shape.as_list()[0]
        return l1_loss

    def smooth_l1loss(self, x):
        l1 = tf.abs(x - self.original)
        smooth_l1loss = tf.reduce_sum(
            0.5 * tf.square(l1) * tf.to_float(tf.abs(l1) < 1) + (tf.abs(l1) - 0.5) * tf.to_float((tf.abs(l1) > 1)))
        # smooth_l1loss = tanh_space(smooth_l1loss)
        return smooth_l1loss

    def fgm_loss(self, x, ori_prob, mfcc_length_out, target, c0=1, c1=0.02, c2=1):
        global original
        preds = ori_prob[0, :, :]

        out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)
        # out_dnn = tanh_space(out_dnn)
        dnn_max = tf.reduce_max(out_dnn, axis=1, keepdims=True)

        targets_index = target

        loss1 = tf.constant(0, dtype=tf.float32)
        targets_index = np.append(targets_index, 0)
        for i in range(0, mfcc_length_out - len(targets_index) - 1):
            targets_index = np.append(targets_index, 91)
        print("~~~", mfcc_length_out, len(targets_index), targets_index)
        lenl = len(targets_index)
        for i in range(0, lenl):
            # loss_delta = tf.maximum(0.0, dnn_max[i] - out_dnn[i, targets_index[i]])
            loss_delta = tf.abs(tf.math.divide((tf.subtract(dnn_max[i], out_dnn[i, targets_index[i]])), dnn_max[i]))
            # loss_delta = tf.maximum(0.0, loss_delta + CONFIDENCE)
            loss1 = tf.add(loss1, loss_delta)

        # loss_l2 = self.l2dist_loss(x)
        loss_l1_smooth = self.smooth_l1loss(x - self.original)
        l1_dist = self.l1_dist_cal(x - self.original)
        l2_dist = self.l2_dist_cal(x - self.original)
        loss_db, vv = self.dbdist_loss(x - self.original)

        # loss = c0 * loss1 + c1 * loss_l1
        loss = loss1
        # loss = loss1
        tmp_result = tf.argmax(out_dnn, axis=1)
        temp_result = tf.reduce_max(x - self.original)
        return -loss, loss1, l1_dist, l2_dist, vv, temp_result

    def Get_grad(self, x, IvectorInput, NOISE=True):
        global original
        w1 = 0.99
        w2 = 1.0 - w1
        x_new = x
        # x_new = x + momentum * tf.sign(self.gradOld_tf)
        # ed = x_new - self.original
        # mask_x = frame_mask(0, self.num_segments_t, self.hop_size, self.seg_size)
        # x_new = x_new + ed * mask_x
        # loss_ods = ods_loss(ori_prob=oriOut, target=targets)
        # q, = tf.gradients(loss_ods, x_new)
        # x_new = x_new + 0.8 * q

        self.mfcc, oriOut, frame_length_out = featrue_prob(x_new, self.num_segments_t, IvectorInput, NOISE)
        # saliency_mask, ind_mask = saliency(self.sig, oriOut, self.targets, self.MAX, self.MIN)  # attention mechanism
        # self.saliency_mask = saliency_mask
        # self.temp_ind_mask = ind_mask

        loss, l1, l1_dist, l2_dist, vv, dnn_max = self.fgm_loss(x=x_new, ori_prob=oriOut,
                                                                mfcc_length_out=frame_length_out, target=self.targets)
        grad, = tf.gradients(loss, x_new)
        # two-stage gradients:loss-->mfcc-->x
        # print("mfcc", mfcc)
        # grad_stage1, = tf.gradients(loss, mfcc)
        # new_mfcc = self.adamOpt_mfcc(mfcc=mfcc, grad=w2*grad_stage1)
        # grad_stage2, = tf.gradients(new_mfcc, x_new)
        # grad_ori, = tf.gradients(mfcc, x_new)
        # grad_all, = tf.gradients(loss,x_new)
        # grad_two, = tf.gradients(l2 + ldb, x_new)
        # new_grad = grad_stage2 *w1
        new_grad = grad
        # self.gradOld_tf = new_grad
        return loss, new_grad, l1, l1_dist, l2_dist, vv, dnn_max

    def loss_grad_XR(self, x_new, NOISE):
        feed = {self.sig: x_new, self.NOISE: NOISE}
        loss, grad, l1, l2, ldb = self.sess.run([self.l, self.g, self.l1, self.l2, self.ldb, ], feed_dict=feed)
        return loss, l1, l2, ldb, grad

    # def write_log(self, fp, bd, xfy, aly, tc, depech, spchb, i, loss_out, l1loss, l2loss, db_loss, snr, psnr):
    #     global args
    #     baidu_reg_result = bd
    #     xfyun_reg_result = xfy
    #     aliyun_reg_result = aly
    #     tencent_reg_result = tc
    #     deepspeech_reg_result = depech
    #     speechbrain_reg_result = spchb
    #
    #     fp.write('%d,\tloss:\t%s,\tl1loss:\t%s,\tl2loss2:\t%s,\tdb_loss2:\t%s,\tsnr:%d\t,\tpsnr:%d\t \n' % (
    #         i, str(loss_out), str(l1loss), str(l2loss), str(db_loss), snr, psnr))
    #     fp.flush()
    #
    #     np.savetxt("zy_0102/mfcc0102/yuxuangf_iter.csv", np.array(self.sig_in), delimiter=" ")
    #     path_wav = wav_write(i, args.value_noise, loss_out, args.iteration_time, args.music_num)
    #
    #     if baidu_reg_result is not None:
    #         result_b, f_name_b = baidu.baidu_recog(path_wav)
    #         print("baidu current result is:", result_b)
    #         baidu_reg_result.write('%s \t %d \t result: %s \n' % (f_name_b, i, result_b))
    #         baidu_reg_result.flush()
    #
    #     if xfyun_reg_result is not None:
    #         result_x, f_name_x = xfyun.xfyun_recog(path_wav)
    #         print("xfyun current result is:", result_x)
    #         xfyun_reg_result.write('%s \t %d \t result: %s \n' % (f_name_x, i, result_x))
    #         xfyun_reg_result.flush()
    #
    #     if aliyun_reg_result is not None:
    #         result_a, f_name_a = aliyun.aliyun_recong(path_wav)
    #         print("aliyun current result is:", result_a)
    #         aliyun_reg_result.write('%s \t %d \t result: %s \n' % (f_name_a, i, result_a))
    #         aliyun_reg_result.flush()
    #
    #     if tencent_reg_result is not None:
    #         result_t, f_name_t = tencentyun.tencent_recogn(path_wav)
    #         print("tencent current result is:", result_t)
    #         tencent_reg_result.write('%s \t %d \t result: %s \n' % (f_name_t, i, result_t))
    #         tencent_reg_result.flush()
    #
    #     if deepspeech_reg_result is not None:
    #         result_d, f_name_d = deepspeech_api.deepspeech_recog(path_wav)
    #         print("deepspeech current result is:", result_d)
    #         deepspeech_reg_result.write('%s \t %d \t result: %s \n' % (f_name_d, i, result_d))
    #         deepspeech_reg_result.flush()
    #
    #     if speechbrain_reg_result is not None:
    #         result_s, f_name_s = SpeechBrain.speechbarin_rcog(path_wav)
    #         print("SpeeechBrain current result is:", result_s)
    #         speechbrain_reg_result.write('%s \t %d \t result: %s \n' % (f_name_s, i, result_s))
    #         speechbrain_reg_result.flush()
    #
    #     if (i == 3999) or ((i > 10) and (math.fabs(loss_out) < 0.5)):
    #         np.savetxt("zy_0102/mfcc0102/yuxuangf_finish.csv", np.array(self.sig_in), delimiter=" ")
    #         path_wav = wav_write_for_libri_speech()
    #         result_b, f_name_b = baidu.baidu_recog(path_wav)
    #         print("baidu current result is:", result_b)
    #         baidu_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_b, i, result_b))
    #         baidu_reg_result.flush()
    #         result_x, f_name_x = xfyun.xfyun_recog(path_wav)
    #         xfyun_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_x, i, result_x))
    #         xfyun_reg_result.flush()
    #         result_a, f_name_a = aliyun.aliyun_recong(path_wav)
    #         aliyun_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_a, i, result_a))
    #         aliyun_reg_result.flush()
    #         return True
    #     else:
    #         return False

    # def run(self):
    #     sess.run(tf.global_variables_initializer())
    #     start = time.time()
    #     print('Start initialize.')
    #     # fp, baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result, deepspeech_reg_result, speechbrain_reg_result = API()
    #     plot = []
    #     snr_record = []
    #     reco_text_l = []
    #     succ_iter = []
    #     Noise = True
    #     var = 0.0
    #     succ_time = 0
    #     loss_stop_record = []
    #     # dnn_out_log = open("./dnn_max.txt", 'w')
    #     for i in range(3000):
    #         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         # mfcc_out = sess.run(self.mfcc, feed_dict={self.sig: sig_in_v})
    #         # np.savetxt("mfcc/mfcc1219/musicfeats_ori.csv", mfcc_out, delimiter=" ")
    #         # os.system("./feats2ivectorinversemfcc1219.sh")
    #         # IvectorInput = np.loadtxt(open("mfcc/mfcc1219/musicivectors_ori.csv", "rb"), delimiter=" ", dtype="float32")
    #         # IvectorInput = np.loadtxt(open("mfcc/musicivectors_ori.csv", "rb"), delimiter=" ", dtype="float32")
    #         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         # ===============================+++++++++++梯度平滑++++++++++++++===================================
    #         grad_m = 0.0
    #         loss_m = 0.0
    #         m_num = args.num_m
    #         # curr_grad = 0.0
    #         sig_in_v = dropout(self.sig_in)
    #
    #         # mfcc_out = sess.run(self.mfcc, feed_dict={self.sig: sig_in_v, self.NOISE: Noise})
    #         # print(sig_in_v)
    #         # print(mfcc_out)
    #         # np.savetxt("/data/guofeng/Speech_Recognition/mfcc/mfcc1219/musicfeats_ori.csv", mfcc_out, delimiter=" ")
    #         # os.system("./feats2ivectorinversemfcc1219.sh > /dev/null 2>&1")
    #         # os.system("./feats2ivectorinversemfcc1219.sh")
    #         # self.IvectorInput = np.loadtxt(
    #         #     open("/data/guofeng/Ensembleattck/kaldi/mfcc/musicivectors_ori.csv", "rb"),
    #         #     delimiter=" ", dtype="float32")
    #         feed = {self.sig: sig_in_v, self.NOISE: Noise}
    #         loss, fgm_grads, pdf_loss, l1_dist, l2_dist, vv, dnn_max_print = self.sess.run(
    #             [self.l, self.g, self.l1, self.l1_dist, self.l2_dist, self.vv, self.dnn_max],
    #             feed_dict=feed)
    #         grad = fgm_grads
    #         self.sess.run([self.delta, self.m, self.v, self.t], feed_dict={self.adam_grad_tf: grad})
    #         # num = sum(dnn_max_print != 3335)
    #         # text = str(i) + 'iter: '+ str(num)+ ' ' + str(dnn_max_print)
    #         # dnn_out_log.write(text)
    #
    #         # for k in range(0, m_num):
    #         #     # sig_in_v = self.sig_in
    #         #     # sig_in_v = k_smooth(sig_in)
    #         #     sig_in_v = dropout(self.sig_in)
    #         #     # sig_in_v = np_input_diversity(self.sig_in)
    #         #     # sig_in_v = 1 / (2 ** k) * self.sig_in
    #         #     feed = {self.sig: sig_in_v, self.NOISE: Noise}
    #         #     loss, fgm_grads, l1loss, l2loss, db_loss, ind, mask, vv = self.sess.run(
    #         #         [self.l, self.g, self.l1, self.l2, self.ldb, self.temp_ind_mask, self.saliency_mask, self.vv],
    #         #         feed_dict=feed)
    #         #     curr_grad = fgm_grads
    #         #     grad_m += fgm_grads
    #         #     loss_m += loss
    #         # average_grad = grad_m / m_num
    #         # average_loss = loss_m / m_num
    #         # grad = 0.9 * self.gradOld_tf + 0.1 * grad  # to momentum way
    #         # self.gradOld_tf = grad
    #         # grad = average_grad
    #         # grad = curr_grad + var
    #         # var = average_grad - curr_grad
    #         # ===============================+++++++++++梯度平滑++++++++++++++===================================
    #         # print("vvvv", vv)
    #         print(
    #             '第', i + 1, '次,', 'smooth its:', m_num, ',loss:', loss, ',pdf_loss:', pdf_loss,
    #             ',l1_dist:%.2f' % l1_dist,
    #             ',l2_dist:%.2f' % l2_dist, 'per_max:%.2f' % dnn_max_print)
    #
    #         # print("self.saliency_mask", np.where(ind == 1))
    #         # print("testing self.m,self.v,self.t:", self.m)
    #         feed_opt = {self.sig: self.sig_in, self.adam_grad_tf: grad}
    #         # self.sig_in, delta = self.sess.run([self.adv, self.delta], feed_dict=feed_opt)
    #         # x_clip, delta_return, delta = self.sess.run([self.x, self.delta_return, self.delta], feed_dict=feed_opt)
    #         # delta_apply = soft_np_clip(delta_return)
    #         # delta_apply = np.pad(delta_apply, (0, self.length - len(delta_apply)), 'constant', constant_values=0.)
    #         # feed_opt2 = {self.sig: self.sig_in, self.delta_apply: delta_return}
    #         self.sig_in = self.sess.run(self.adv, feed_dict=feed_opt)
    #         # print("delta:", delta, np.sum(delta), "gap:", self.sig_in - self.original)
    #         # snr = np_SNR(self.original, self.sig_in) # 原始音频做分子 扰动做分母
    #         # snr_2 = np_SNR_2(self.original, self.sig_in) # 对抗样本做分子 扰动做分母
    #         # p_snr = np_PSNR(self.sig_in, self.original)
    #         # print("snr_right %.2f"%(snr))
    #         # snr_record.append(snr)
    #         plot.append(-loss)
    #         # np.savetxt("zy_0102/mfcc0102/yuxuangf_iter.csv", np.array(self.sig_in), delimiter=" ")
    #         wav_write(i + 1, loss, 1, 1, self.sig_in)
    #         # if i % 10 == 0:
    #         #     FT = self.write_log(fp, baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result,
    #         #                         deepspeech_reg_result, speechbrain_reg_result, i + ii, average_loss, l1loss, l2loss,
    #         #                         db_loss, snr, p_snr)
    #         #     if FT is True:
    #         #         break
    #         if np.abs(pdf_loss) <= 0.1 and i % 10 == 0:
    #             temp_save_path = "/data/guofeng/Speech_Recognition/recogn/temp_save.wav"
    #             wav_write_for_ae(temp_save_path, self.sig_in)
    #             os.system('./recogn/each_audio.sh %s' % temp_save_path)
    #             result = subprocess.getoutput("./recogn/decode_speech.sh")
    #             text_list = result.split('\n')
    #             for line in text_list:
    #                 if line.startswith("temp_save"):
    #                     print(line)
    #                     reco_text = line.strip(' ').split(' ')[1:]
    #
    #             # fp = open('./recogn/decode_result.txt', mode='w')
    #             # fp.write(result)
    #             # fp.close()
    #             # os.system("cat ./recogn/decode_result.txt | grep '^temp_save' | tee ./recogn/name-text2.txt")
    #             # os.system("cat ./recogn/decode_result.txt | grep '^[0-9]' | tee ./recogn/name-text2.txt")
    #             # os.system("cat decode_result.txt | grep '^4epoch' | tee name-text2.txt ")
    #             # os.system("cat decode_result.txt | grep '^[0-9]\+epoch' | tee name-text2.txt ")
    #             # os.system(
    #             #     "cat ./recogn/name-text2.txt|  awk '{for(i = 2; i <= NF; i++) printf(\"%s \", $i);printf(\"\\n\")}' > ./recogn/text2.txt")
    #             single_cer = cer(reco_text, "okay google call nine one one".split(' '))
    #             # cer_list, mean_cer = get_cer("./recogn/text2.txt", "okay google call nine one one")
    #             if single_cer == 0:
    #                 succ_iter.append(i + 1)
    #         if np.abs(loss) <= 3:
    #             succ_time += 1
    #             if succ_time >= 5:
    #                 break
    #     np.savetxt("zy_0102/mfcc0102/yuxuangf_finish.csv", np.array(self.sig_in), delimiter=" ")
    #     path_wav = wav_write_for_libri_speech()
    #     print("adv.wav has finish")
    #     pl(plot)
    #     pl_snr(snr_record)
    #
    #     end = time.time()
    #     print("Attack time is:", end - start)
    #     print(succ_iter)
    def run(self):
        sess.run(tf.global_variables_initializer())
        start = time.time()
        print('Start initialize.')
        fp, baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result, \
        deepspeech_reg_result, speechbrain_reg_result = API()
        plot = []
        Noise = True
        var = 0.0
        for i in range(4000):
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # mfcc_out = sess.run(mfcc, feed_dict={sig: sig_in})
            # np.savetxt("mfcc/mfcc1219/musicfeats_ori.csv", mfcc_out, delimiter=" ")
            # os.system("./feats2ivectorinversemfcc1219.sh")
            # IvectorInput = np.loadtxt(open("mfcc/mfcc1219/musicivectors_ori.csv", "rb"), delimiter=" ", dtype="float32")
            # IvectorInput = np.loadtxt(open("mfcc/musicivectors_ori.csv", "rb"), delimiter=" ", dtype="float32")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ===============================+++++++++++梯度平滑++++++++++++++===================================
            grad_m = 0.0
            loss_m = 0.0
            m_num = args.num_m
            curr_grad = 0.0
            for k in range(0, m_num):
                # sig_in_v = self.sig_in
                # sig_in_v = k_smooth(sig_in)
                sig_in_v = dropout(self.sig_in)
                # sig_in_v = np_input_diversity(self.sig_in)
                # sig_in_v = 1 / (2 ** k) * self.sig_in
                feed = {self.sig: sig_in_v, self.NOISE: Noise}
                loss, fgm_grads, l1loss, l2loss, db_loss, ind, mask, vv = self.sess.run(
                    [self.l, self.g, self.l1, self.l2, self.ldb, self.temp_ind_mask, self.saliency_mask, self.vv],
                    feed_dict=feed)
                curr_grad = fgm_grads
                grad_m += fgm_grads
                loss_m += loss
            # average_grad = grad_m / m_num
            average_grad = grad_m / np.mean(np.abs(grad_m))  # 'np.mean(np.abs())' important Large impact on results
            average_loss = loss_m / m_num
            grad = momentum * self.gradOld_tf + average_grad  # to momentum way  Good
            self.gradOld_tf = grad
            # grad = average_grad
            # grad = average_grad + var / np.mean(np.abs(average_grad + var))
            # var = average_grad - curr_grad
            # ===============================+++++++++++梯度平滑++++++++++++++===================================
            print("itr: %d,smmoth its: %d,loss:%01f,l1loss:%01f,l2loss:%01f,db_loss:%01f" % (
                i, m_num, average_loss, l1loss, l2loss, db_loss))

            snr = np_SNR(self.sig_in, self.original)
            p_snr = np_PSNR(self.sig_in, self.original)
            print("snr and p_snr:%f %f" % (snr, p_snr))
            # print("self.saliency_mask", np.where(ind == 1))
            # print("testing self.m,self.v,self.t:", self.m)

            feed_opt = {self.sig: self.sig_in, self.adam_grad_tf: grad}
            # self.sig_in, delta = self.sess.run([self.adv, self.delta], feed_dict=feed_opt)
            # x_clip, delta_return, delta = self.sess.run([self.x, self.delta_return, self.delta], feed_dict=feed_opt)
            # delta_apply = soft_np_clip(delta_return)
            # delta_apply = np.pad(delta_apply, (0, self.length - len(delta_apply)), 'constant', constant_values=0.)
            # feed_opt2 = {self.sig: self.sig_in, self.delta_apply: delta_return}
            self.sig_in = self.sess.run(self.adv, feed_dict=feed_opt)
            # print("delta:", delta, np.sum(delta), "gap:", self.sig_in - self.original)

            plot.append(-average_loss)
            if i % 10 == 0:
                FT = self.write_log(fp, baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result,
                                    deepspeech_reg_result, speechbrain_reg_result, i, average_loss, l1loss, l2loss,
                                    db_loss, snr, p_snr)
                if FT is True:
                    break
        print("adv.wav has finish")
        pl(plot)
        end = time.time()
        print("Attack time is:", end - start)


if __name__ == '__main__':
    if args.iteration_time == 1:
        (rate, sig_in_wav) = wav.read(args.music_path)
        print(args.music_path)
    else:
        (rate, sig_in_wav) = wav.read("csv_to_wav/adversarial.wav")
    original = sig_in_wav
    with tf.Session() as sess:
        print('Start initalize.')

        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer(['sal_mask']))
        # genetic_instance = genetic(sig_in=sig_in_wav, sess=sess, input_wave=sig_in_wav, mfcc_shape=None)
        # writer = tf.summary.FileWriter("E://TensorBoard//graph", sess.graph)
        # writer.close()dddddd
        # itr = genetic_instance.run_genetic()
        # genetic_instance.run(0)
        atl = Attack(sig_in=sig_in_wav, sess=sess, mfcc_shape=None)
        atl.run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # When done, ask the threads to stop.
    coord.request_stop()
    # fp.close()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
