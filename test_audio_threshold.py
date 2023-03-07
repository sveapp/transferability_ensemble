# coding=utf-8
import librosa

import sys
sys.settrace
import numpy as np
import generate_masking_threshold as generate_mask
from absl import flags
import generate_masking_threshold
import scipy.io.wavfile as wav

flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string(
    "input", "read_data.txt", "Input audio .wav file(s), at 16KHz (separated by spaces)"
)

# data processing
flags.DEFINE_integer("window_size", "2048", "window size in spectrum analysis")
flags.DEFINE_integer(
    "max_length_dataset",
    "223200",
    "the length of the longest audio in the whole dataset",
)
flags.DEFINE_float(
    "initial_bound", "2000", "initial l infinity norm for adversarial perturbation"
)

# training parameters
flags.DEFINE_string("checkpoint", "./model/ckpt-00908156", "location of checkpoint")
flags.DEFINE_integer("batch_size", "1", "batch size")
flags.DEFINE_float("lr_stage1", "100", "learning_rate for stage 1")
flags.DEFINE_float("lr_stage2", "1", "learning_rate for stage 2")
flags.DEFINE_integer("num_iter_stage1", "1000", "number of iterations in stage 1")
flags.DEFINE_integer("num_iter_stage2", "4000", "number of iterations in stage 2")
flags.DEFINE_integer("num_gpu", "0", "which gpu to run")

FLAGS = flags.FLAGS


def ReadFromWav(data, batch_size):
    """
    data form is np[3,1]--[[path,文本,文本]
    Returns:
        audios_np: a numpy array of size (batch_size, max_length) in float
        trans: a numpy array includes the targeted transcriptions (batch_size, )
        th_batch: a numpy array of the masking threshold, each of size (?, 1025)
        psd_max_batch: a numpy array of the psd_max of the original audio (batch_size)
        max_length: the max length of the batch of audios
        sample_rate_np: a numpy array
        masks: a numpy array of size (batch_size, max_length)
        masks_freq: a numpy array of size (batch_size, max_length_freq, 80)
        lengths: a list of the length of original audios
    """
    audios = []
    lengths = []
    th_batch = []
    psd_max_batch = []  # 准备计算掩蔽阈值

    # read the .wav file
    for i in range(batch_size):
        sample_rate_np, audio_temp = wav.read(FLAGS.root_dir + str(data[0, i]))
        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp

        length = len(audio_np)

        audios.append(audio_np)
        lengths.append(length)

    max_length = max(lengths)

    # pad the input audio
    audios_np = np.zeros([batch_size, max_length])
    masks = np.zeros([batch_size, max_length])
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    max_length_freq = max(lengths_freq)
    masks_freq = np.zeros([batch_size, max_length_freq, 80])
    for i in range(batch_size):
        audio_float = audios[i].astype(float)
        audios_np[i, : lengths[i]] = audio_float
        masks[i, : lengths[i]] = 1
        masks_freq[i, : lengths_freq[i], :] = 1

        # compute the masking threshold
        th, psd_max = generate_mask.generate_th(
            audios_np[i], sample_rate_np, FLAGS.window_size
        )
        th_batch.append(th)
        psd_max_batch.append(psd_max)

    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)

    # read the transcription
    trans = data[2, :]

    return (
        audios_np,
        trans,
        th_batch,
        psd_max_batch,
        max_length,
        sample_rate_np,
        masks,
        masks_freq,
        lengths,
    )


if __name__ == "__main__":
    audio_path = '/data/guofeng/audio_adversaril/cleverhans-master/cleverhans_v3.1.0/examples/adversarial_asr/LibriSpeech/test-clean/61/70968/61-70968-0011.wav'
    y, sr = librosa.load(audio_path)
    rate ,data= wav.read(audio_path)
    data = data.astype(float)
    print("audio:", len(y), sr)

    theta_xs, psd_max = generate_masking_threshold.generate_th(y, 16000, 2048)
    print("theta_xs", theta_xs, type(theta_xs), theta_xs.shape)
    print("psd_max", psd_max)

    theta_xs, psd_max = generate_masking_threshold.generate_th(data,16000, 2048)
    print("theta_xs", theta_xs, type(theta_xs), theta_xs.shape)
    print("psd_max", psd_max)
