# -*- coding:utf-8 -*-
import os
import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile as wav
import wave
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sox
import tensorflow as tf
#
# tf = tf.compat.v1
# tf.disable_v2_behavior()

epsilon = 1e-08


def energy_tensor(x, shape):
    # signal_shape = tf.convert_to_tensor(shape,tf.float32)
    signal_shape = tf.cast(shape,tf.float32)
    e = tf.sqrt(tf.math.divide(tf.reduce_sum(tf.math.square(x)), signal_shape))
    return e


def energy_np(x, signal_shape):
    e = np.sqrt(np.divide(np.sum(np.square(x)), signal_shape))
    return e


def log10(x):
    numerator = tf.log(x + epsilon)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def np_SNR(origianl_waveform, target_waveform):  # 整体信噪比
    # 单位 dB
    signal = np.sum(origianl_waveform ** 2)
    noise_per = np.sum((origianl_waveform - target_waveform) ** 2)
    snr = 10. * np.log10(np.abs(signal / (noise_per + epsilon)))
    if np.isnan(snr) or np.isinf(snr): snr = 1000
    return snr


def tf_SNR(origianl_waveform, target_waveform):  # 整体信噪比
    # 单位 dB
    signal = tf.reduce_sum(origianl_waveform ** 2)
    noise_per = tf.reduce_sum((origianl_waveform - target_waveform) ** 2)
    snr = 10. * (tf.math.log(signal) / (tf.math.log(noise_per) + epsilon))
    return snr


def np_PSNR(origianl_waveform, target_waveform):  # 峰值信噪比
    MAX = np.max(target_waveform)
    MSE = np.mean((origianl_waveform - target_waveform) ** 2)
    psnr = 20. * np.log10(np.divide(MAX, np.sqrt(MSE) + epsilon))
    if np.isinf(psnr):
        psnr = 1000.0
    if np.isnan(psnr):
        psnr = 1000.0
    return psnr


def tf_PSNR(origianl_waveform, target_waveform):  # 峰值信噪比
    MAX = tf.reduce_max(target_waveform)
    MSE = tf.reduce_mean((origianl_waveform - target_waveform) ** 2)
    psnr = 20. * (tf.math.log(MAX) / (tf.math.log(MSE) + epsilon))
    return psnr


def clip(x_np, lg):  # 裁剪
    length = lg
    clip_w = np.random.randint(0, 1, size=[1], dtype=np.int32)[0]

    if (clip_w + length) <= len(x_np):
        clip_ret = x_np[clip_w: clip_w + length]
    else:
        clip_ret = np.pad(x_np, (0, (clip_w + length) - len(x_np)), constant_values=0.)
        clip_ret = clip_ret[:length]
    return clip_ret


def clip_norm(x):
    mean, variance = tf.nn.moments(x=x, axes=0)
    xx = (x - mean) / variance
    return xx


def clip_mean(x, a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    means = tf.reduce_sum(x) / tf.cast(tf.count_nonzero(x), tf.float32)
    xx = (x - means) / (means - a)
    return xx


def roll(x_np, max_ratio=0.01):  # 循环平移
    frame_num = x_np.shape[0]
    max_shifts = frame_num * max_ratio  # around 0.1% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts, size=[1], dtype=np.int32)[0]
    return np.roll(x_np, shift=nb_shifts)


def pitch_shifting(x_np, sr=8000, n_steps=5, octave=12):
    return librosa.effects.pitch_shift(float(x_np), sr, n_steps, octave)


def speed_librosa(x_np, min_speed=0.9, max_speed=1.):
    """
    librosa时间拉伸
    :param x_np: 音频数据，一维
    :param max_speed: 不要低于0.9，太低效果不好
    :param min_speed: 不要高于1.1，太高效果不好
    :return:
    """
    samples = x_np.copy()  # framebuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = np.random.uniform(min_speed, max_speed, size=[1])[0]
    samples = samples.astype(np.float32)
    samples = librosa.effects.time_stretch(samples, speed)
    samples = samples.astype(data_type)
    return samples


def speed_numpy(x_np, min_speed=0.9, max_speed=1.1):
    """
    线形插值速度增益
    :param x_np: 音频数据，一维
    :param max_speed: 不能低于0.9，太低效果不好
    :param min_speed: 不能高于1.1，太高效果不好
    :return:
    """
    samples = x_np.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = np.random.uniform(min_speed, max_speed, size=[1])[0]
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def db_chage(wav_path, updown_db, outwav_path):
    sound = AudioSegment.from_file(wav_path, "wav")  # 加载WAV文件
    db = sound.dBFS  # 取得WAV文件的声音分贝值
    normalized_sound = match_target_amplitude(sound, db + updown_db)  # db+10表示比原来的声音大10db,需要加大音量就加多少，反之则减多少
    normalized_sound.export(outwav_path, format="wav")
    return outwav_path


def updown_noise_db(x_wav, db, NR):
    """
    just we using the function ,when NR is true ,means noise up,db >0,else NR is false,means db<0
    @param x_wav:
    @param db:
    @param NR:
    @return:
    """
    global original
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
        temp_wav_path = wavwrite_for_noise(temp_wav, delta_value_fixed)  # original wav
        db = db  # reduce or up db 20
        out_temp = 'ori_audio/temp22.wav'
        pa1 = db_chage(temp_wav_path, db, out_temp)  # reduce original wav db
        (rate_temp, delta_value_fixed) = wav.read(pa1)
        x = delta_noise + delta_value_fixed
    return x


def wavwrite_for_noise(wav_write_path, new_wav):
    RATE = 8000
    # GENERATE MONO FILE #
    wv = wave.open(wav_write_path, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = new_wav
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wav_write_path


def wavwrite_for_librispeech():
    RATE = 8000
    # GENERATE MONO FILE #
    wavpath = "ori_audio/fix_test.wav"
    wv = wave.open(wavpath, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = np.loadtxt(open("../zy_0102/mfcc0102/fix_test", "r"), delimiter=",", dtype="float32")
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wavpath


def oscillogram_spectrum(audio_path):
    """
    画出音频文件audio_path的声波图和频谱图
    :param audio_path:音频文件路径
    :return:
    """
    # 读取wav文件
    filename = audio_path
    wavefile = wave.open(filename, 'r')  # open for writing
    # 读取wav文件的四种信息的函数。期中numframes表示一共读取了几个frames。
    nchannels = wavefile.getnchannels()
    sample_width = wavefile.getsampwidth()
    framerate = wavefile.getframerate()
    numframes = wavefile.getnframes()
    print("channel", nchannels)
    print("sample_width", sample_width)
    print("framerate", framerate)
    print("numframes", numframes)
    # 建一个y的数列，用来保存后面读的每个frame的amplitude。
    y = np.zeros(numframes)
    # for循环，readframe(1)每次读一个frame，取其前两位，是左声道的信息。右声道就是后两位啦。
    # unpack是struct里的一个函数，简单说来就是把＃packed的string转换成原来的数据，无论是什么样的数据都返回一个tuple。这里返回的是长度为一的一个
    # tuple，所以我们取它的第零位。
    for i in range(numframes):
        val = wavefile.readframes(1)
        left = val[0:2]
        # right = val[2:4]
        v = struct.unpack('h', left)[0]
        y[i] = v
    # framerate就是声音的采用率，文件初读取的值。
    Fs = framerate
    time = np.arange(0, numframes) * (1.0 / framerate)
    # 显示时域图(波形图)
    plt.subplot(211)
    plt.plot(time, y)
    # 显示频域图(频谱图)
    plt.subplot(212)
    plt.specgram(y, NFFT=1024, Fs=Fs, noverlap=900)
    plt.show()


def librosa_plot(path):
    x, sr = librosa.load(path, sr=8000)
    print("shape and rate:", x.shape, sr)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()  # 横轴为时间，纵轴为幅度（amplitude）

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))  # 把幅度转成分贝格式
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()  # 这里横轴是时间，纵轴是频率，颜色则代表分贝（声音的响度），可以看到越红的地方信号音量越大

    n0 = 0
    n1 = x.shape[0]
    plt.figure(figsize=(14, 5))
    plt.plot(x[n0:n1])
    plt.grid()
    plt.show()  # 过零率作为音频信号的重要特征，用于判断该音频信号中是否包括期望的有用信号，或该音频信号中只有噪声
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    print(sum(zero_crossings))

    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)
    # Displaying the MFCCs:
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()


def add_silent(path, path_out):
    music = AudioSegment.from_wav(path)
    silent = AudioSegment.silent(duration=5000)
    music_silent = silent + music
    music_silent.export(path_out, format='wav')


def resample_8k(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y_8k = librosa.resample(y, sr, 8000)
    out = audio_path + "8k.wav"
    librosa.output.write_wav(out, y_8k, 8000)
    return out


def resample_wav(file, rate):
    tfm = sox.Transformer()
    tfm.rate(rate)
    out_path = file.split('.wav')[0] + "_hr.wav"
    tfm.build(file, out_path)
    return out_path


if __name__ == '__main__':
    music_path = '../ori_audio/sig_wav.wav'
    (rate, sig_in) = wav.read(music_path)
    print("below zero:", sig_in[sig_in < 0])
    original = sig_in
    # music_path = 'csv_to_wav/adv/adversarial_yuxuan(ok).wav'
    # music_path = "ori_audio/fix_test.wav"
    # music_path = 'csv_to_wav/adv/adv_audio/aliyunAudio.wav'
    # =============================plot==========================================
    oscillogram_spectrum('../ori_audio/gh.wav')
    librosa_plot(music_path)
    # =============================plot==========================================
    # =============================音量==========================================
    # sound = AudioSegment.from_file(music_path, "wav")  # 加载WAV文件
    # db = sound.dBFS  # 取得WAV文件的声音分贝值
    # normalized_sound = match_target_amplitude(sound, db + 20)  # db+10表示比原来的声音大10db,需要加大音量就加多少，反之则减多少
    # normalized_sound.export("ori_audio/re_out_db_ok.wav", format="wav")

    (rate1, sig_in1) = wav.read('../ori_audio/gh.wav')
    outpath = "ori_audio/re_out_db_ok.wav"
    okx = updown_noise_db(sig_in1, -8, True)
    pd = wavwrite_for_noise('../ori_audio/sig_wav_re_noise.wav', okx)
    print("okx:", okx)
    # opa = db_chage(music_path, 20, outpath)
    # ==============================音量=========================================

    # =============================add silent==========================================
    out_path = '../ori_audio/good_time_silent.wav'
    add_silent(music_path, out_path)
    # =============================add silent==========================================
    # =============================write wav==========================================
    (rate, sig_in) = wav.read(music_path)
    sig_in_test = 1 / 2 * sig_in
    w_path = '../ori_audio/good_time_write.wav'
    pa = wavwrite_for_noise(w_path, sig_in_test)

    # # sig_in_test = 1 / 4 * sig_in
    # sig_in_test = 1 / 16 * sig_in
    # np.savetxt("zy_0102/mfcc0102/fix_test", np.array(sig_in_test), delimiter=" ")
    # test = wavwrite_for_librispeech()
    # (rate, sig_in) = wav.read('ori_audio/good_time.wav')
    # (rate1, sig_in1) = wav.read('csv_to_wav/adv/adv_audio/aliyunAudio.wav')
    # feat_sa = SpecAugment()
    # data_input = feat_sa.run(sig_in, rate)
    # print("rate audio", rate, len(sig_in), sig_in)
    # # np.savetxt('good_time_sig', sig_in, delimiter='')
    # np.savetxt('aliyun_sig', sig_in1, delimiter='')
    # np.savetxt('aliyun_sig', sig_in - sig_in1, delimiter='')
    # print("SpecAugment audio", rate, len(data_input), data_input)
    # =============================write wav==========================================
    # =============================resample==========================================

    out = resample_8k('../ori_audio/man.wav')
    y1, r1 = librosa.load(out, sr=None)
    print(y1, r1)

    y0, sr0 = librosa.load('../ori_audio/good_time.wav', sr=None)
    print("resample8k:", len(y0))
    y, sr = librosa.load('../ori_audio/back.wav8k.wav', sr=None)
    print("resample8k:", len(y))
    y1, sr1 = librosa.load('../ori_audio/man.wav8k.wav', sr=None)
    print("resample8k:", len(y1))

    (rate, sig_in) = wav.read('../ori_audio/good_time.wav')
    print("librosa.load:", y, len(y), sr)
    print("scipy.io.wavfile:", rate, len(sig_in), sig_in, max(sig_in), np.mean(sig_in))
    # print("audio sr:", sr)
    # =============================resample==========================================
    # ============================read WAV==============================================================

    f = wave.open('../ori_audio/good_time.wav', 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
    print("wave:", waveData)
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    # ============================read WAV==============================================================
    # audio_loader = get_default_audio_adapter()
    # sample_rate = 8000
    # waveform, _ = audio_loader.load('en_outfile.wav', sample_rate=sample_rate)
    # # Perform the separation :
    # prediction = separator.separate(waveform)
