# coding=utf-8
import math
import os
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
import librosa
import struct
import wave
import warnings
from shutil import copyfile
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse

from kaldi.audio.audio_test import log10, energy_tensor
from tf_logits import get_logits
import generate_masking_threshold as generate_mask

sys.path.append("DeepSpeech")

from lingvo import model_registry

from tool import Transform, create_features, create_inputs
from lingvo import model_imports
import generate_masking_threshold as generate_mask
import time
from lingvo.core import cluster_factory

from kaldi.transferability_check.aliyun import aliyun
from kaldi.transferability_check.baidu import baidu
from kaldi.transferability_check.tencent import tencentyun
from kaldi.transferability_check.xfyun import xfyun
from kaldi.ori_yuxuangf import add_noise, dropout, tanh_space, featrue_prob

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

print(tf.test.is_gpu_available())

toks = " abcdefghijklmnopqrstuvwxyz'-"  # 26  #26个字母
CONFIDENCE = 0.0
EPS = 1.0
K = 1.0
momentum = 1.0
boxmin = 0.
boxmax = 1.
epsilon = 1e-08
boxmul = (boxmax - boxmin) / 2.  # 0.5
boxplus = (boxmin + boxmax) / 2.  # 0.5

path = ['./attack_sample/good_time.wav']
filename = './attack_sample/good_time.wav'.split('/')[-1]
out = ['./attack_sample/defense_adv' + '/adv_' + filename]
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--in', type=str, dest="input", nargs='+', default=path, help="Input audio")
parser.add_argument('--target', type=str, default="okay google turn off the light", help="Target transcription")
parser.add_argument('--out', type=str, nargs='+', default=out, required=False, help="Path for the adv")
parser.add_argument('--outprefix', type=str, required=False, help="Prefix of path for adversarial examples")
parser.add_argument('--finetune', type=str, nargs='+', required=False, help=".wav file(s) to use as a starting point")
parser.add_argument('--lr', type=int, required=False, default=100, help="Learning rate for optimization")
parser.add_argument('--iterations', type=int, required=False, default=1000, help="Maximum number of iterations ")
parser.add_argument('--l2penalty', type=float, required=False, default=float('inf'), help="Weight for l2 penalty")
parser.add_argument('--mp3', action="store_const", const=True, required=False, help="MP3 compression resistant adv")
parser.add_argument('--restore_path', type=str,
                    default='/data/guofeng/Ensembleattck/deepspeech-0.4.1-checkpoint/model.v0.4.1',
                    help="Path to the ds -ctc checkpoint (ending in model0.4.1)")

parser.add_argument('--value_noise', type=int, default=4000, help="Input scale noise ")
parser.add_argument('--sigma', type=int, default=4000, help="Input value of normal's noise ")
parser.add_argument('--noise_class', type=str, default='uniform', help="class")
parser.add_argument('--iteration_time', required=False, type=int, default=1, help="iteration_time")
parser.add_argument('--music_path', type=str,
                    default='/data/guofeng/Ensembleattck/kaldi/ori_audio/good_time.wav', help="music_path")
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

parser.add_argument("--root_dir", type=str, default='./', help="location of Librispeech")
parser.add_argument("--input", type=str, default='command.txt', help="Input audio .wav file(s)")
parser.add_argument('--window_size', type=int, default=2048, help="window size in spectrum analysis")
parser.add_argument("--max_length_dataset", type=int, default=223200, help="the length of the longest")
parser.add_argument("--initial_bound", type=float, default=2000, help="infinity norm")
parser.add_argument("--checkpoint", type=str, default="./model/ckpt-00908156", help="location checkpoint")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--lr_stage1", type=float, default=100, help="learning_rate for stage 1")
parser.add_argument("--lr_stage2", type=float, default=1, help="learning_rate for stage 2")
parser.add_argument("--num_iter_stage1", type=int, default=80, help="number of iterations in stage 1")
parser.add_argument("--num_iter_stage2", type=int, default=3000, help="number of iterations in stage 2")
parser.add_argument("--num_gpu", type=int, default=0, help="which gpu to run")
FLAGS = parser.parse_args()


def ReadFromWav(path, batch_size, trans):
    """
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
    psd_max_batch = []

    # read the .wav file
    for i in range(batch_size):
        sample_rate_np, audio_temp = wav.read(path)

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
        print(FLAGS.window_size)
        th, psd_max = generate_mask.generate_th(audios_np[i], sample_rate_np, FLAGS.window_size)
        th_batch.append(th)
        psd_max_batch.append(psd_max)

    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)

    # read the transcription
    trans = trans

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


def API():
    baidu_reg_result = open('./check/baidu.txt', 'w')
    # xfyun_reg_result = open('transferability_check/xfyun/xfyun.txt', 'w')
    # aliyun_reg_result = open('transferability_check/aliyun/aliyun.txt', 'w')
    xfyun_reg_result = None
    aliyun_reg_result = None
    tencent_reg_result = open('./check/tencent.txt', 'w')
    return baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result


def write_wav(iteration):
    RATE = 8000

    wavpath = "./itr_wav/sample%s.wav" % iteration
    wv = wave.open(wavpath, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = np.loadtxt(open("./itr_wav/sample.csv", "r"), delimiter=",", dtype="float32")
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wavpath


class DS_Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None):

        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size, 1), dtype=np.float32), name='qq_phrase_lengths')
        self.apply_delta = tf.clip_by_value(delta, -8000, 8000) * self.rescale
        self.new_input = new_input = self.apply_delta * mask + original
        noise = tf.random_normal(new_input.shape, stddev=2)  # (1,51072)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)
        self.logits = logits = get_logits(pass_in, lengths)
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=lengths)
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input - self.original) ** 2, axis=1) + l2penalty * ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
        elif loss_fn == "CW":
            raise NotImplemented(
                "The current version of this project does not include the CW loss function implementation.")
        else:
            raise
        self.loss = loss
        self.ctcloss = ctcloss

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.grad, var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(2 * self.grad, var)])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        sess.run(tf.variables_initializer(new_vars + [delta]))

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, audio, lengths, target, finetune=None, itr_save=False, path=None, out_path=None):
        sess = self.sess

        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.mask.assign(
            np.array([[1 if iii < lll else 0 for iii in range(self.max_audio_len)] for lll in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if ioi < lol else 0 for ioi in range(self.phrase_length)]
                                              for lol in (np.array(lengths) - 1) // 320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        # target:list(1):[[20, 8, 9, 19, 0, 9, 19, 0, 1, 0, 20, 5, 19, 20]]
        sess.run(self.target_phrase.assign(np.array([list(t) + [0] * (self.phrase_length - len(t)) for t in target])))

        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size, 1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None] * self.batch_size
        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune - audio))

        # We'll make a bunch of iterations of gradient descent here  梯度下降开始
        now = time.time()
        plot_loss1 = []
        plot_loss2 = []

        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            now = time.time()

            # Print out some debug information every 10 iterations. 输出一些调试的信息
            if i % 10 == 0:
                sess.run((self.new_input, self.delta, self.decoded, self.logits))
                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]

                for out, logits in lst:
                    out[0].values
                    res = np.zeros(out[0].dense_shape) + len(toks) - 1  # res = {ndarray: (1, 43)} [28..[28]]
                    for ii in range(len(out[0].values)):
                        x, y = out[0].indices[ii]
                        res[x, y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]
                    print("res is:", "\n".join(res))

                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits, axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(loss - 1) // 320]) for y, loss in zip(res2, lengths)]
                    print("res2:", "\n".join(res2))

            feed_dict = {}
            delta, el, ctcloss, loss, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                                       self.ctcloss, self.loss,
                                                                       self.logits, self.new_input,
                                                                       self.train),
                                                                      feed_dict)
            plot_loss1.append(ctcloss)
            plot_loss2.append(loss)
            print("%.3f" % np.mean(ctcloss), "\t", "\t".join("%.3f" % x for x in ctcloss))  # %.3f保留3位小数，\t表示同行

            np.argmax(logits, axis=2).T
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (self.loss_fn == "CTC" and i % 10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                        or (i == MAX - 1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii] * 2000 > np.max(np.abs(delta)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(delta[ii])) / 2000.0)
                        rescale[ii] = np.max(np.abs(delta[ii])) / 2000.0  # 此处d代替了delta，rescale

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss=%f bound=%f" % (ii, ctcloss[ii], 2000 * rescale[ii][0]))
                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("/tmp/adv.wav", 8000,
                              np.array(np.clip(np.round(new_input[ii]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    np.savetxt('./attack_sample/adv.txt', new_input)

            if itr_save is True:
                name = str(i) + '_' + path[0].split('/')[-1]
                path1 = out_path + '/' + name
                wav.write(path1, 8000,
                          np.array(np.clip(np.round(new_input[ii][:lengths[ii]]), -2 ** 15, 2 ** 15 - 1),
                                   dtype=np.int16))

        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(plot_loss1, label="$loss$")
        plt.plot(plot_loss2, label="$l$")
        plt.legend()
        plt.show()
        return final_deltas


class LG_Attack:
    def __init__(
            self,
            sess,
            input_max_len,
            batch_size=1,  # 5
            lr_stage1=100,
            lr_stage2=0.1,  # 1.0
            num_iter_stage1=1000,
            num_iter_stage2=4000,
            th=None,
            psd_max_ori=None,
    ):
        self.sess = sess
        self.input_max_len = input_max_len
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.batch_size = batch_size
        self.lr_stage1 = lr_stage1

        tf.set_random_seed(1234)
        params = model_registry.GetParams("asr.librispeech.Librispeech960Wpm", "Test")
        # print(params)
        params.random_seed = 1234
        params.is_eval = True
        params.cluster.worker.gpus_per_replica = 1
        cluster = cluster_factory.Cluster(params.cluster)
        with cluster, tf.device(cluster.GetPlacer()):
            model = params.cls(params)

            # placeholders
            self.input_tf = tf.placeholder(tf.float32, shape=[batch_size, None], name="qq_input")  # (5,?)
            self.tgt_tf = tf.placeholder(tf.string)
            self.sample_rate_tf = tf.placeholder(tf.int32, name="qq_sample_rate")
            self.th = tf.placeholder(tf.float32, shape=[batch_size, None, None], name="qq_th")  # (5, ?,? )
            self.psd_max_ori = tf.placeholder(tf.float32, shape=[batch_size], name="qq_psd")  # (5,)
            self.mask = tf.placeholder(dtype=np.float32, shape=[batch_size, None], name="qq_mask")
            self.mask_freq = tf.placeholder(dtype=np.float32, shape=[batch_size, None, 80])  # (5,?,80)
            self.noise = tf.placeholder(np.float32, shape=[batch_size, None], name="qq_noise")  # (5,?)
            self.maxlen = tf.placeholder(np.int32)
            self.lr_stage2 = tf.placeholder(np.float32)

            # variable
            # self.delta_large = tf.Variable(np.zeros((batch_size, FLAGS.max_length_dataset), dtype=np.float32),
            #                                name="qq_delta")
            self.delta_large = tf.Variable(np.zeros((batch_size, self.input_max_len), dtype=np.float32),
                                           name="qq_delta")
            self.rescale = tf.Variable(np.ones((batch_size, 1), dtype=np.float32), name="qq_rescale")  # (5,1)
            self.alpha = tf.Variable(np.ones(batch_size, dtype=np.float32) * 0.05, name="qq_alpha")  # (5,)

            self.delta = tf.slice(tf.identity(self.delta_large), [0, 0], [batch_size, self.maxlen])  # (5,?)
            self.apply_delta = (
                    tf.clip_by_value(self.delta, -FLAGS.initial_bound, FLAGS.initial_bound) * self.rescale)  # (5,?)
            self.new_input = self.apply_delta * self.mask + self.input_tf  # (5,?)
            self.pass_in = tf.clip_by_value(self.new_input + self.noise, -(2 ** 15), 2 ** 15 - 1)  # (5,?)

            self.features = create_features(self.pass_in, self.sample_rate_tf, self.mask_freq)  # (5,?,80)
            self.inputs = create_inputs(model, self.features, self.tgt_tf, self.batch_size, self.mask_freq)

            task = model.GetTask()
            metrics = task.FPropDefaultTheta(self.inputs)

            self.celoss = tf.get_collection("per_loss")[0]
            self.decoded = task.Decode(self.inputs)

        self.loss_th_list = []
        self.transform = Transform(FLAGS.window_size)
        for i in range(self.batch_size):
            logits_delta = self.transform((self.apply_delta[i, :]), self.psd_max_ori[i])
            loss_th = tf.reduce_mean(tf.nn.relu(logits_delta - self.th[i]))
            loss_th = tf.expand_dims(loss_th, dim=0)
            self.loss_th_list.append(loss_th)
        self.loss_th = tf.concat(self.loss_th_list, axis=0)

        self.optimizer1 = tf.train.AdamOptimizer(self.lr_stage1)
        self.optimizer2 = tf.train.AdamOptimizer(self.lr_stage2)

        self.grad1, var1 = self.optimizer1.compute_gradients(self.celoss, [self.delta_large])[0]  # all (5,223200)
        self.grad21, var21 = self.optimizer2.compute_gradients(self.celoss, [self.delta_large])[0]
        self.grad22, var22 = self.optimizer2.compute_gradients(self.alpha * self.loss_th, [self.delta_large])[0]

        self.train1 = self.optimizer1.apply_gradients([(tf.sign(self.grad1), var1)])
        self.train21 = self.optimizer2.apply_gradients([(self.grad21, var21)])
        self.train22 = self.optimizer2.apply_gradients([(self.grad22, var22)])


class KL_Attack:
    def __init__(self, sess, sig_in):
        """
        @type sess: object
        """
        self.original = sig_in
        self.sig_in = sig_in
        self.MAX = np.max(sig_in)
        self.MIN = np.min(sig_in)
        self.sess = sess
        self.h0_frame = 28
        self.n_feature = 220
        self.n_pdfid = 8629
        self.n = 50
        self.FS = 8000
        self.seg_size = self.FS * 0.025
        self.hop_size = self.FS * 0.01
        self.num_segments_t = math.floor((len(self.sig_in) - self.seg_size) / self.hop_size) + 1
        self.length = len(self.sig_in)
        self.targets = np.loadtxt(open(args.target_path, "rb"), delimiter=" ", dtype="int16")
        self.IvectorInput = np.loadtxt(open(args.ivector_path, "rb"), delimiter=" ", dtype="float32")
        self.input_shape = self.length

        self.sig = tf.placeholder(tf.float32, shape=self.length)
        self.adam_grad_tf = tf.placeholder(tf.float32, shape=self.input_shape)
        self.delta_apply = tf.placeholder(tf.float32, shape=self.input_shape)
        self.NOISE = tf.placeholder(tf.bool)

        zero = tf.zeros(self.input_shape, dtype=tf.float32)
        self.m = tf.get_variable(name='m', dtype=tf.float32, initializer=zero)
        self.v = tf.get_variable(name='v', dtype=tf.float32, initializer=zero)
        self.t = tf.get_variable(name='t', dtype=tf.float32, initializer=1.0)
        self.delta = tf.get_variable(name='delta', dtype=tf.float32, initializer=zero)

        self.gradOld_tf = np.zeros(shape=self.input_shape, dtype=np.float32)
        self.saliency_mask = tf.ones_like(self.sig, dtype=tf.float32)
        self.temp_ind_mask = tf.ones_like(self.sig, dtype=tf.float32)

        self.l, self.g, self.l1, self.l2, self.ldb, self.vv = self.Get_grad(self.sig, self.NOISE)
        self.adv = self.adamOpt(x=self.sig, grad=self.adam_grad_tf)

    def adamOpt(self, x, grad, lr=100, beta1=0.9, beta2=0.999, epsilon=1e-08, clip_min=-32767, clip_max=32767):
        m = self.m
        v = self.v
        t = self.t + 1
        # t = self.t
        grads = grad

        lr_t = lr * tf.sqrt(1 - tf.pow(beta2, t)) / (1 - tf.pow(beta1, t))
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * tf.square(grads)
        pertuabations = tf.math.ceil(lr_t * m / (tf.sqrt(v) + epsilon))
        pertuabations_fixed = (momentum * self.delta + pertuabations)
        x = x + pertuabations_fixed

        delta_value = x - self.original
        delta_value_fixed = tf.clip_by_value(delta_value, -8000, 8000)
        x = self.original + delta_value_fixed
        if (clip_min is not None) and (clip_max is not None):
            x = tf.clip_by_value(x, clip_min, clip_max)

        #  you must understand tf.assign and '=' assignment statement you must use tf.assign to change a variable
        self.delta = tf.assign(self.delta, pertuabations_fixed)
        self.m = tf.assign(self.m, m)
        self.v = tf.assign(self.v, v)
        self.t = tf.assign(self.t, t)

        return x

    def dbdist_loss(self, x):  # bug is input 'tanh_space' parameter x can't 0
        put_noise = tf.random.normal(shape=(len(self.original),), mean=0, stddev=0.1)
        delta_v = tanh_space(x + put_noise) - tanh_space(self.original)
        dbdist_loss = 20.0 * log10(energy_tensor(delta_v, len(self.original)))
        tt = self.delta
        return dbdist_loss, tt

    def l2dist_loss(self, x):
        # l2dist = tf.reduce_sum(tf.square(tanh_space(x) - tanh_space(self.original)))
        l2dist = tf.reduce_sum(tf.square(x - self.original)) * 1 / x.shape[0]
        return l2dist

    def l1_loss(self, x):
        l1_loss = tf.reduce_sum(tf.abs(x - self.original))
        return l1_loss

    def smooth_l1loss(self, x):
        l1 = tf.abs(x - self.original)
        smooth_l1loss = tf.reduce_sum(
            0.5 * tf.square(l1) * tf.to_float(tf.abs(l1) < 1) + (tf.abs(l1) - 0.5) * tf.to_float((tf.abs(l1) > 1)))
        # smooth_l1loss = tanh_space(smooth_l1loss)
        return smooth_l1loss

    def fgm_loss(self, x, ori_prob, target, c0=1, c1=0.02, c2=1):
        preds = ori_prob[0, :, :]
        out_dnn = tf.convert_to_tensor(preds, dtype=tf.float32)
        dnn_max = tf.reduce_max(out_dnn, axis=1, keepdims=True)
        lenl = len(target)
        targets_index = tf.convert_to_tensor(target, tf.int32)
        loss1 = tf.constant(0, dtype=tf.float32)

        for i in range(0, lenl):
            k = targets_index[i]
            # loss_delta = tf.maximum(0.0, dnn_max[i] - tf.gather_nd(out_dnn, [i, targets_index[i]]) + CONFIDENCE)
            # loss_delta = tf.maximum(0.0, dnn_max[i] - out_dnn[i, targets_index[i]] + CONFIDENCE)
            # loss_delta = tf.abs(tf.math.divide((tf.subtract(dnn_max[i], tf.gather_nd(out_dnn, [[i, k]]))), dnn_max[i]))
            loss_delta = tf.abs(tf.math.divide((tf.subtract(dnn_max[i], out_dnn[i, targets_index[i]])), dnn_max[i]))
            loss_delta = tf.maximum(0.0, loss_delta)
            loss1 = tf.add(loss1, loss_delta)

        loss_l1_smooth = self.smooth_l1loss(x)
        loss_l1 = self.l1_loss(x)
        loss_db, vv = self.dbdist_loss(x)
        loss = loss1
        return -loss, -loss1, -loss_l1, -loss_db, vv

    def Get_grad(self, x, NOISE=True):
        w1 = 0.99
        w2 = 1.0 - w1
        x_new = x

        mfcc, oriOut = featrue_prob(x_new, self.num_segments_t, self.IvectorInput, NOISE)
        loss, l1, l2, ldb, vv = self.fgm_loss(x=x_new, ori_prob=oriOut, target=self.targets)
        grad, = tf.gradients(loss, x_new)
        new_grad = grad
        return loss, new_grad, l1, l2, ldb, vv


class Ens:
    def __init__(self, sess, input_audio):

        self.sess = sess
        self.input_shape = input_audio.shape
        self.input_audio = input_audio

        self.model_ds = DS_Attack(sess, 'CTC', len(args.target), max_audio_len=len(input_audio[-1]), batch_size=1,
                                  mp3=args.mp3, learning_rate=args.lr, num_iterations=args.iterations,
                                  l2penalty=args.l2penalty, restore_path=args.restore_path)
        print ("deepspeech model loaded over....")
        self.model_aspire = KL_Attack(self.sess, self.input_audio[-1])
        print("aspire model loaded over....")
        self.model_lingvo = LG_Attack(sess, input_max_len=len(input_audio[-1]), batch_size=FLAGS.batch_size,
                                      lr_stage1=FLAGS.lr_stage1,
                                      lr_stage2=FLAGS.lr_stage2, num_iter_stage1=FLAGS.num_iter_stage1,
                                      num_iter_stage2=FLAGS.num_iter_stage2)
        print ("lingvo model loaded over....")

        zero = tf.zeros(self.input_shape, dtype=tf.float32)
        self.me = tf.get_variable(name='me', dtype=tf.float32, initializer=zero)
        self.ve = tf.get_variable(name='ve', dtype=tf.float32, initializer=zero)
        self.te = tf.get_variable(name='te', dtype=tf.float32, initializer=1.0)
        self.deltae = tf.get_variable(name='deltae', dtype=tf.float32, initializer=zero)

        self.place_input_audio = tf.placeholder(tf.float32, shape=self.input_shape)
        self.place_grad = tf.placeholder(tf.float32, shape=self.input_shape)
        # self.place_delta_apply_grad = tf.placeholder(tf.float32, shape=input_audio)

        self.noise = add_noise(shape_in=self.input_shape, mu=0, best_sig=None, RO=False)  # add noise

        self.loss_ds = self.model_ds.loss
        self.loss_as = self.model_aspire.l
        self.loss_lg1 = self.model_lingvo.celoss
        self.loss_lg2 = self.model_lingvo.loss_th

        self.decode_ds = self.model_ds.decoded
        self.decode_lg = self.model_lingvo.decoded
        self.logist_ds = self.model_ds.logits

        # self.loss_avg = (self.loss_ds + self.loss_as + self.loss_lg1) / 3.0

        self.grad_ds = self.model_ds.grad
        self.grad_as = self.model_aspire.g
        self.grad_lg1 = self.model_lingvo.grad21
        self.grad_lg2 = self.model_lingvo.grad22
        # self.grad_avg = (self.grad_ds + self.grad_as + self.grad_lg1) / 3.0

        self.adv = self.adam_opt(self.place_input_audio, self.place_grad)

    def adam_opt(self, x, grad, lr=100, beta1=0.9, beta2=0.999, epsilon=1e-08, clip_min=-32767, clip_max=32767):
        me = self.me
        ve = self.ve
        te = self.te + 1
        # t = self.t
        grads = grad

        lr_t = lr * tf.sqrt(1 - tf.pow(beta2, te)) / (1 - tf.pow(beta1, te))
        me = beta1 * me + (1 - beta1) * grads
        ve = beta2 * ve + (1 - beta2) * tf.square(grads)
        pertuabations = tf.math.ceil(lr_t * me / (tf.sqrt(ve) + epsilon))  # 向上取整
        pertuabations_fixed = (1.0 * self.deltae + pertuabations)
        x = x + pertuabations_fixed
        if (clip_min is not None) and (clip_max is not None):
            x = tf.clip_by_value(x, clip_min, clip_max)

        #  you must understand tf.assign and '=' assignment statement you must use tf.assign to change a variable 'self.delta = pertuabations_fixed'
        self.delta = tf.assign(self.deltae, pertuabations_fixed)
        self.me = tf.assign(self.me, me)
        self.ve = tf.assign(self.ve, ve)
        self.te = tf.assign(self.te, te)
        return x

    def write_log(self, bd, xfy, aly, tc, i, loss_out):

        baidu_reg_result = bd
        xfyun_reg_result = xfy
        aliyun_reg_result = aly
        tencent_reg_result = tc

        np.savetxt("./itr_wav/sample.csv", self.input_audio, delimiter=" ")
        path_wav = write_wav(i)

        if baidu_reg_result is not None:
            result_b, f_name_b = baidu.baidu_recog(path_wav)
            print("baidu current result is:", result_b)
            baidu_reg_result.write('%s \t %d \t result: %s \n' % (f_name_b, i, result_b))
            baidu_reg_result.flush()

        if xfyun_reg_result is not None:
            result_x, f_name_x = xfyun.xfyun_recog(path_wav)
            print("xfyun current result is:", result_x)
            xfyun_reg_result.write('%s \t %d \t result: %s \n' % (f_name_x, i, result_x))
            xfyun_reg_result.flush()

        if aliyun_reg_result is not None:
            result_a, f_name_a = aliyun.aliyun_recong(path_wav)
            print("aliyun current result is:", result_a)
            aliyun_reg_result.write('%s \t %d \t result: %s \n' % (f_name_a, i, result_a))
            aliyun_reg_result.flush()

        if tencent_reg_result is not None:
            result_t, f_name_t = tencentyun.tencent_recogn(path_wav)
            print("tencent current result is:", result_t)
            tencent_reg_result.write('%s \t %d \t result: %s \n' % (f_name_t, i, result_t))
            tencent_reg_result.flush()

        if (i == 3999) or ((i > 10) and (math.fabs(loss_out) < 0.5)):
            np.savetxt("./itr_wav/sample.csv", np.array(self.input_audio), delimiter=" ")
            result_b, f_name_b = baidu.baidu_recog(path_wav)
            print("baidu current result is:", result_b)
            baidu_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_b, i, result_b))
            baidu_reg_result.flush()
            result_x, f_name_x = xfyun.xfyun_recog(path_wav)
            xfyun_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_x, i, result_x))
            xfyun_reg_result.flush()
            result_a, f_name_a = aliyun.aliyun_recong(path_wav)
            aliyun_reg_result.write('finish \t%s \t %d \t result: %s \n' % (f_name_a, i, result_a))
            aliyun_reg_result.flush()
            return True
        else:
            return False

    def run_attack(self, audio, target, lengths, trans, sample_rate, th_batch, psd_max_batch, masks, masks_freq, maxlen,
                   lr_stage2, finetune=None, itr_save=False, path=None, out_path=None):

        sess = self.sess
        # sess.run(tf.global_variables_initializer())
        # =========================================ds-model======================================================
        sess.run(tf.variables_initializer([self.model_ds.delta]))
        sess.run(self.model_ds.original.assign(np.array(audio)))
        sess.run(self.model_ds.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.model_ds.mask.assign(
            np.array([[1 if iii < lll else 0 for iii in range(self.model_ds.max_audio_len)] for lll in lengths])))
        sess.run(
            self.model_ds.cwmask.assign(np.array([[1 if ioi < lol else 0 for ioi in range(self.model_ds.phrase_length)]
                                                  for lol in (np.array(lengths) - 1) // 320])))
        sess.run(self.model_ds.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.model_ds.target_phrase.assign(
            np.array([list(t) + [0] * (self.model_ds.phrase_length - len(t)) for t in target])))
        c = np.ones((self.model_ds.batch_size, self.model_ds.phrase_length))
        sess.run(self.model_ds.importance.assign(c))
        sess.run(self.model_ds.rescale.assign(np.ones((self.model_ds.batch_size, 1))))
        final_deltas = [None] * self.model_ds.batch_size
        if finetune is not None and len(finetune) > 0:
            sess.run(self.model_ds.delta.assign(finetune - audio))
        # ========================================aspire-model======================================================
        self.model_aspire.sess.run(tf.global_variables_initializer())
        # =========================================lingvo-model======================================================
        self.model_lingvo.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)
        sess.run(tf.assign(self.model_lingvo.rescale, np.ones((self.model_lingvo.batch_size, 1), dtype=np.float32)))
        # sess.run(tf.assign(self.model_lingvo.delta_large,
        #                    np.zeros((self.model_lingvo.batch_size, FLAGS.max_length_dataset), dtype=np.float32), ))
        sess.run(tf.assign(self.model_lingvo.delta_large,
                           np.zeros((self.model_lingvo.batch_size, self.model_lingvo.input_max_len),
                                    dtype=np.float32), ))
        noise = np.zeros(audio.shape)
        noise = np.random.uniform(0.0, 1.0, audio.shape)
        # =========================================api-check======================================================
        baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result = API()

        # =========================================run-attack======================================================
        print("input audio:", audio.shape)
        for i in range(4000):
            feed_dict1 = {}
            sig_in_v = np.reshape(audio, audio.shape[-1])
            # sig_in_v = dropout(self.input_audio)
            # sig_in_v = 1 / (2 ** k) * self.sig_in
            feed_dict2 = {self.model_aspire.sig: sig_in_v, self.model_aspire.NOISE: True}
            feed_dict3 = {
                self.model_lingvo.input_tf: audio,
                self.model_lingvo.tgt_tf: trans,
                self.model_lingvo.sample_rate_tf: sample_rate,
                self.model_lingvo.th: th_batch,
                self.model_lingvo.psd_max_ori: psd_max_batch,
                self.model_lingvo.mask: masks,
                self.model_lingvo.mask_freq: masks_freq,
                self.model_lingvo.noise: noise,
                self.model_lingvo.maxlen: maxlen,
                self.model_lingvo.lr_stage2: lr_stage2,
            }

            loss_ds, grad_ds = sess.run([self.loss_ds, self.grad_ds], feed_dict1)
            loss_as, grad_as = sess.run([self.loss_as, self.grad_as], feed_dict2)
            loss_lg, grad_lg1 = sess.run([self.model_lingvo.celoss, self.grad_lg1], feed_dict3)
            print("loss_lg", loss_lg, loss_lg.shape)

            grad_ds = np.reshape(grad_ds, (1, audio.shape[-1]))
            grad_as = np.reshape(grad_as, (1, audio.shape[-1]))
            grad_lg1 = np.reshape(grad_lg1, (1, audio.shape[-1]))

            print("grad:", grad_ds.shape, grad_as.shape, type(grad_as), grad_lg1.shape, type(grad_lg1))
            loss_avg = (loss_ds + loss_as + loss_lg) / 3.0
            grad_avg = (2 * grad_ds + grad_as + grad_lg1) / 3.0
            print(" grad_avg :", grad_avg.shape)

            feed_dict_adv = {self.place_input_audio: audio, self.place_grad: grad_avg}
            audio = sess.run(self.adv, feed_dict_adv)
            print('avg loss:%02f,loss_ds:%02f,loss_as:%02f,loss_lg:%02f' % (loss_avg, loss_as, loss_as, loss_lg))

            if i % 10 == 0:
                self.input_audio = audio
                pre_ds, logist = sess.run([self.decode_ds, self.logist_ds], feed_dict1)
                lst = [(pre_ds, logist)]
                for out, logits in lst:
                    out[0].values
                    res = np.zeros(out[0].dense_shape) + len(toks) - 1  # res = {ndarray: (1, 43)} [28..[28]]
                    for ii in range(len(out[0].values)):
                        x, y = out[0].indices[ii]
                        res[x, y] = out[0].values[ii]
                    res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]
                    print("Deepspeech model decode:", "\n".join(res))
                decode_as = "unable"
                pre_lg = sess.run(self.decode_lg, feed_dict3)

                print("aspire model decode", decode_as)
                print("lingvo model decode:", type(pre_lg), pre_lg["topk_decoded"][0, 0])
                _ = self.write_log(baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result, i,
                                   loss_avg)


def main(input_audio_path, target):
    (audio, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, length) \
        = ReadFromWav(input_audio_path, 1, trans=target)
    trans = np.array([trans.upper()], dtype='|S125')
    print("trans:", trans, trans.shape, type(trans))
    with tf.device("/cpu:0"):
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:
            attacker = Ens(sess, audio)
            attacker.run_attack(audio=audio, lengths=length, target=[[toks.index(x) for x in args.target]] * 1,
                                trans=trans, sample_rate=sample_rate,
                                th_batch=th_batch, psd_max_batch=psd_max_batch, masks=masks, masks_freq=masks_freq,
                                maxlen=maxlen, lr_stage2=FLAGS.lr_stage2)


if __name__ == '__main__':
    input_audio_path = './audio_carry/4s/01_English_0070.wav'
    target = "okay google turn off the light"
    main(input_audio_path=input_audio_path, target=target)
