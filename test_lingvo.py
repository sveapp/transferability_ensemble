# coding=utf-8
import os

import sys

sys.settrace
from lingvo import model_registry
from lingvo import model_imports
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import create_features, create_inputs
from lingvo.core import cluster_factory
from absl import flags
from absl import app

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置为1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出

import faulthandler

faulthandler.enable()
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

print(tf.test.is_gpu_available())
# data directory
flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string(
    "input", "testaudio.txt", "Input audio .wav file(s), at 16KHz (separated by spaces)"
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
        path = './' + str(data[0, i])
        print(i, path)
        sample_rate_np, audio_temp = wav.read(path)
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
        masks[i, : lengths[i]] = 19
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


def main(unused_argv):
    params = model_registry.GetParams("asr.librispeech.Librispeech960Wpm","Test")
    # print(params)

    params.cluster.worker.gpus_per_replica = 1
    params.random_seed = 1234
    params.is_eval = True
    params.cluster.worker.gpus_per_replica = 1
    cluster = cluster_factory.Cluster(params.cluster)
    #
    model = params.cls(params)
    # print("model:", model)

    # task = model.GetTask()
    # print("task:", task)

    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])  # 10
    batch_size = FLAGS.batch_size  # 5
    num_loops = int(num / batch_size)  # 2
    # assert num % batch_size == 0

    # with tf.device("/cpu:54"):
    # tfconf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session() as sess:
        with sess.as_default():
            for l in range(num_loops):
                data_sub = data[:, l * batch_size: (l + 1) * batch_size]
                (
                    audios,
                    trans,
                    th_batch,
                    psd_max_batch,
                    maxlen,
                    sample_rate,
                    masks,
                    masks_freq,
                    lengths,
                ) = ReadFromWav(data_sub, batch_size)

            sess.run(tf.initializers.global_variables())
            # saver = tf.train.import_meta_graph("./model/ckpt-00908156.meta", clear_devices=True)
            graph = tf.get_default_graph()

            saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
            saver.restore(sess, FLAGS.checkpoint)
            # variable_names = [v.name for v in tf.trainable_variables()]
            # variable_names = [v.name for v in tf.global_variables()]
            # values = sess.run(variable_names)
            # i = 0
            # for k, v in zip(variable_names, values):
            #     i += 1
            #     if k.find('encode') != -1:
            #         print("第 {i} 个variable",i)
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print(v)
            # graph = tf.get_default_graph()
            # all_ops = graph.get_operations()
            # for el in all_ops:
            #     print(el.name)
            checkpoint_path = os.path.join("./model", "ckpt-00908156")
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("tensor_name: ", key)

            a = tf.GraphKeys.UPDATE_OPS
            c = tf.get_collection(a)
            print("collection", a)
            print("collection", c)

            audios = np.float32(audios)
            # trans = np.float32(trans)
            th_batch = np.float32(th_batch)
            psd_max_batch = np.float32(psd_max_batch)
            maxlen = np.float32(maxlen)
            # sample_rate = np.float32(sample_rate)
            masks = np.float32(masks)
            masks_freq = np.float32(masks_freq)
            lengths = np.float32(lengths)

            new_input = audios
            tf.cast(new_input, tf.float32)
            pass_in = tf.clip_by_value(new_input, -(2 ** 15), 2 ** 15 - 1)
            tf.cast(pass_in, tf.float32)
            features = create_features(pass_in, sample_rate, masks_freq)
            tf.cast(features, tf.float32)
            inputs = create_inputs(model, features, trans, batch_size, masks_freq)
            # tf.cast(inputs,tf.float32)
            # print("input:", inputs)

            task = model.GetTask()
            # print("task:", task)

            m = task.FPropDefaultTheta(inputs)

            # print("task2:", m)
            # self.celoss with the shape (batch_size)

            celoss = tf.get_collection("per_loss")[0]
            # print("loss:", celoss)

            predictions = task.Decode(inputs)
            # print("decoder:", predictions)

            # sd = tf.Session()

            c, d = sess.run([celoss, predictions])
            # d = sess.run( predictions)
            print("c d:", c)

            for i in range(batch_size):
                print(
                    "example: {}, loss: {}".format(
                        num_loops * batch_size + i, c[0]
                    )
                )
                print("pred:{}".format(d["topk_decoded"][i, 0]))
                print("targ:{}".format(trans[i].lower()))
                print("true: {}".format(data[1, i].lower()))


if __name__ == '__main__':
    app.run(main)
