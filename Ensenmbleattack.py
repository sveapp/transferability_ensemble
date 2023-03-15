# coding=utf-8
import math
import random
import sys
import os
import tensorflow as tf

from generate_imperceptible_adv import FLAGS
import generate_imperceptible_adv as LINGVO_MODEL
import generate_masking_threshold as generate_mask

from kaldi import ori_yuxuangf as ASPIRE
from kaldi.ori_yuxuangf import add_noise, dropout
from kaldi.ori_yuxuangf import args as args_kl

import attack as DS
from attack import args as args_ds

from kaldi.transferability_check.aliyun import aliyun
from kaldi.transferability_check.baidu import baidu
from kaldi.transferability_check.tencent import tencentyun
from kaldi.transferability_check.xfyun import xfyun

import numpy as np
import scipy.io.wavfile as wav

import wave
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
toks = " abcdefghijklmnopqrstuvwxyz'-"  # 26  #26个字母


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
        print(sample_rate_np, audio_temp)

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


def ds_model(sess, max_len):
    modl_ds = DS.Attack(sess, 'CTC', len(args_ds.target), max_audio_len=max_len,
                        batch_size=1,
                        mp3=args_ds.mp3,
                        learning_rate=args_ds.lr,
                        num_iterations=args_ds.iterations,
                        l2penalty=args_ds.l2penalty,
                        restore_path=args_ds.restore_path
                        )
    return modl_ds


def lingvo_model(sess, max_len):
    model_lingvo = LINGVO_MODEL.Attack(sess,
                                       input_max_len=max_len,
                                       batch_size=FLAGS.batch_size,
                                       lr_stage1=FLAGS.lr_stage1,
                                       lr_stage2=FLAGS.lr_stage2,
                                       num_iter_stage1=FLAGS.num_iter_stage1,
                                       num_iter_stage2=FLAGS.num_iter_stage2,
                                       )
    return model_lingvo


def aspire_model(sess, sig_in):
    print("aspire_model input hsape", sig_in.shape)
    model_aspire = ASPIRE.Attack(sess=sess, sig_in=sig_in, mfcc_shape=None)
    return model_aspire


def API():
    baidu_reg_result = open('./check/baidu.txt', 'w')
    # xfyun_reg_result = open('transferability_check/xfyun/xfyun.txt', 'w')
    # aliyun_reg_result = open('transferability_check/aliyun/aliyun.txt', 'w')
    xfyun_reg_result = None
    aliyun_reg_result = None
    tencent_reg_result = open('./check/tencent.txt', 'w')
    return baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result


def write_wav(iteration, loss):
    RATE = 16000

    wavpath = "./itr_wav/sample%s_loss%s.wav" % (iteration, loss)
    wv = wave.open(wavpath, 'wb')
    wv.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

    wvData = np.loadtxt(open("./itr_wav/sample.csv", "r"), delimiter=",", dtype="float")
    wvData = wvData.astype(np.int16)
    wv.writeframes(wvData.tobytes())
    wv.close()
    return wavpath


class Ens:
    def __init__(self, sess, input_audio):

        self.sess = sess
        self.input_shape = input_audio.shape
        self.input_audio = input_audio
        self.original = input_audio

        print("input_audio.", input_audio.shape)
        self.model_ds = ds_model(sess=self.sess, max_len=self.input_shape[-1])
        print("deepspeech model loaded over....")
        self.model_aspire = aspire_model(sess=self.sess, sig_in=self.input_audio[-1])
        print("aspire model loaded over....")
        self.model_lingvo = lingvo_model(sess=self.sess, max_len=self.input_shape[-1])
        print("lingvo model loaded over....")

        zero = tf.zeros(self.input_shape, dtype=tf.float32)
        self.me = tf.get_variable(name='me', dtype=tf.float32, initializer=zero)
        self.ve = tf.get_variable(name='ve', dtype=tf.float32, initializer=zero)
        self.te = tf.get_variable(name='te', dtype=tf.float32, initializer=1.0)
        self.deltae = tf.get_variable(name='deltae', dtype=tf.float32, initializer=zero)

        self.place_input_audio = tf.placeholder(tf.float32, shape=self.input_shape)
        self.place_grad = tf.placeholder(tf.float32, shape=self.input_shape)
        self.place_delta = tf.placeholder(tf.float32, shape=self.input_shape)

        self.noise = add_noise(shape_in=self.input_shape, mu=0, best_sig=None, RO=False)  # add noise

        self.loss_ds = self.model_ds.loss
        self.loss_as = self.model_aspire.l
        self.loss_lg_ce = self.model_lingvo.celoss
        self.loss_lg_th = self.model_lingvo.loss_th

        self.decode_ds = self.model_ds.decoded
        self.decode_lg = self.model_lingvo.decoded
        self.logist_ds = self.model_ds.logits

        # self.loss_avg = (self.loss_ds + self.loss_as + self.loss_lg1) / 3.0

        self.delta_ds = self.model_ds.apply_delta
        self.delta_as = self.model_aspire.apply_delta
        self.delta_lg = self.model_lingvo.apply_delta
        self.grad_ds = self.model_ds.grad
        self.grad_as = self.model_aspire.g
        self.grad_lg21 = self.model_lingvo.grad21
        self.grad_lg22 = self.model_lingvo.grad22
        # self.grad_avg = (self.grad_ds + self.grad_as + self.grad_lg1) / 3.0

        # self.adv = self.adam_opt(self.place_input_audio, self.place_grad)
        self.adv = self.original + self.place_delta  # 这样做的原因主要是，对x求导和对delta求导无法合并在一块

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

        np.savetxt("./itr_wav/sample.csv", np.array(self.input_audio[-1], dtype=np.float32), delimiter=" ")
        path_wav = write_wav(i, loss_out)

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
            np.savetxt("./itr_wav/sample.csv", np.array(self.input_audio[-1]), delimiter=" ")
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

        global audio_new, feed_dict3, feed_dict1
        sess = self.sess
        #  =========================================ds-model======================================================
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
        ds_vars = self.model_ds.global_var
        gl_vars = tf.global_variables()
        init_vars = [x for x in gl_vars if x not in ds_vars]
        sess.run(tf.variables_initializer(init_vars))

        # =========================================lingvo-model======================================================
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
        audio_new = audio
        momentum = 1.0
        gradOld_ds, gradOld_as, gradOld_lg = 0., 0., 0.
        lossOld_ds, lossOld_as, lossOld_lg = 0., 0., 0.
        deltaOld_ds, deltaOld_as, deltaOld_lg = 0., 0., 0.
        for i in range(4000):
            m = 5
            grad_ds_all, grad_as_all, grad_lg_all = 0, 0, 0
            loss_ds_all, loss_as_all, loss_lg_all = 0, 0, 0
            for j in range(m):
                feed_dict1 = {}
                sig_in_v = np.reshape(audio_new, audio_new.shape[-1])
                sig_in_v = dropout(sig_in_v)
                # sig_in_v = 1 / (2 ** j) * sig_in_v
                feed_dict2 = {self.model_aspire.sig: sig_in_v, self.model_aspire.NOISE: True}
                feed_dict3 = {  # music audio can't alart audio
                    # self.model_lingvo.input_tf: audio_new + noise,  # 得严格注意这里的参数的传递的情况，表示为original
                    self.model_lingvo.input_tf: audio + noise,
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
                sess.run([self.model_aspire.delta, self.model_aspire.m, self.model_aspire.v, self.model_aspire.t],
                         feed_dict={self.model_aspire.adam_grad_tf: grad_as})
                loss_lg_ce, grad_lg21 = sess.run([self.loss_lg_ce, self.grad_lg21], feed_dict3)
                grad_ds_all += grad_ds
                grad_as_all += grad_as
                grad_lg_all += grad_lg21

                loss_ds_all += loss_ds
                loss_as_all += loss_as
                loss_lg_all += loss_lg_ce

            grad_ds_avg = grad_ds_all / np.mean(np.abs(grad_ds_all))
            grad_as_avg = grad_as_all / np.mean(np.abs(grad_as_all))
            grad_lg_avg = grad_lg_all / np.mean(np.abs(grad_lg_all))
            grad_ds2 = momentum * gradOld_ds + grad_ds_avg  # to momentum way  Good使用动量梯度
            grad_as2 = momentum * gradOld_as + grad_as_avg  # to momentum way  Good使用动量梯度
            grad_lg2 = momentum * gradOld_lg + grad_lg_avg  # to momentum way  Good使用动量梯度
            gradOld_ds, gradOld_as, gradOld_lg = grad_ds2, grad_as2, grad_lg2

            loss_ds_avg = loss_ds_all / m
            loss_as_avg = loss_as_all / m
            loss_lg_avg = loss_lg_all / m

            loss_all = np.abs(loss_ds_avg) + np.abs(loss_as_avg) + np.abs(loss_lg_avg)
            w1 = np.abs(loss_ds_avg) / loss_all
            w2 = np.abs(loss_as_avg) / loss_all
            w3 = np.abs(loss_lg_avg) / loss_all

            ## The method of stochastic gradient ensemble
            r = random.randint(1, 3)
            new_grad = grad_as2 if r == 1 else (grad_ds2 if r == 2 else grad_lg2)

            ## The method of dynamic weight ensemble
            # grad_as2 = np.reshape(grad_as2, (1, grad_as2.shape[0]))
            # w_a = np.exp(-np.linalg.norm(grad_as2, ord=2))
            # w_d = np.exp(-np.linalg.norm(grad_ds2, ord=2))
            # w_l = np.exp(-np.linalg.norm(grad_lg2, ord=2))
            # w_all = w_a + w_d + w_l
            # w_a, w_d, w_l = w_a / w_all, w_d / w_all, w_l / w_all
            # new_grad = grad_ds2 * w_d + grad_as2 * w_a + grad_lg2 * w_l

            new_grad_ds2 = new_grad
            new_grad_as2 = np.reshape(new_grad, (new_grad.shape[1],))
            new_grad_lg2 = new_grad

            feed_dict11 = {self.model_ds.place_grad_ds: new_grad_ds2}
            feed_dict22 = {self.model_aspire.sig: np.reshape(audio_new, audio_new.shape[-1]),
                           self.model_aspire.NOISE: True,
                           self.model_aspire.adam_grad_tf: new_grad_as2}
            feed_dict33 = {self.model_lingvo.input_tf: audio,
                           self.model_lingvo.tgt_tf: trans,
                           self.model_lingvo.sample_rate_tf: sample_rate,
                           self.model_lingvo.th: th_batch,
                           self.model_lingvo.psd_max_ori: psd_max_batch,
                           self.model_lingvo.mask: masks,
                           self.model_lingvo.mask_freq: masks_freq,
                           self.model_lingvo.noise: noise,
                           self.model_lingvo.maxlen: maxlen,
                           self.model_lingvo.lr_stage2: lr_stage2,
                           self.model_lingvo.place_grad_lg1: new_grad_lg2,
                           self.model_lingvo.place_grad_lg21: new_grad_lg2}

            delta_ds, adv_ds, _ = sess.run([self.delta_ds, self.model_ds.new_input, self.model_ds.train], feed_dict11)
            delta_as, adv_as = sess.run([self.delta_as, self.model_aspire.adv], feed_dict22)
            delta_lg, adv_lg, _ = sess.run([self.delta_lg, self.model_lingvo.new_input, self.model_lingvo.train1],
                                           feed_dict33)

            delta_ds = np.reshape(delta_ds, (1, audio_new.shape[-1])) + momentum * deltaOld_ds
            delta_as = np.reshape(delta_as, (1, audio_new.shape[-1])) + momentum * deltaOld_as
            delta_lg = np.reshape(delta_lg, (1, audio_new.shape[-1])) + momentum * deltaOld_lg
            deltaOld_ds, deltaOld_as, deltaOld_lg = delta_ds, delta_as, delta_lg  # 多重梯度向量

            if np.abs(loss_ds_avg) < 8:
                w1 = 0.1
            if np.abs(loss_as_avg) > 50:
                w2 = 1.0
            if np.abs(loss_lg_avg) < 0.1:
                w3 = 0.0
            loss_avg = np.abs(loss_ds_avg) * w1 + np.abs(loss_as_avg) * w2 + np.abs(loss_lg_avg) * w3

            delta2 = (delta_ds + delta_as + delta_lg) / 3
            feed_dict_adv = {self.place_delta: delta2}
            audio_new = sess.run(self.adv, feed_dict_adv)

            print('%d,avg loss:%02f,loss_ds:%02f,loss_as:%02f,loss_lg:%02f' % (
                i, loss_avg, loss_ds_avg, loss_as_avg, loss_lg_avg))

            if i % 10 == 0:
                self.input_audio = audio_new
                pre_ds, logist = sess.run([self.decode_ds, self.logist_ds], feed_dict1)
                lst = [(pre_ds, logist)]
                for out, logits in lst:
                    res = np.zeros(out[0].dense_shape) + len(toks) - 1
                    for ii in range(len(out[0].values)):
                        x, y = out[0].indices[ii]
                        res[x, y] = out[0].values[ii]
                res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]
                decode_as = "unable"
                pre_lg = sess.run(self.decode_lg, feed_dict3)
                print("Deepspeech model decode:", "\n".join(res))
                print("aspire model decode", decode_as)
                print("lingvo model decode:", type(pre_lg), pre_lg["topk_decoded"][0, 0])
                _ = self.write_log(baidu_reg_result, xfyun_reg_result, aliyun_reg_result, tencent_reg_result, i,
                                   loss_avg)


def main(input_audio_path, target):
    (audio, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, length) \
        = ReadFromWav(input_audio_path, 1, trans=target)
    trans = np.array([trans.upper()], dtype='|S125')  # lingvo model target need a ndm

    with tf.device("/cpu:0"):
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:
            attacker = Ens(sess, audio)
            # attacker.model_ds.attack(audio, length,[[toks.index(x) for x in args_ds.target]] * 1,itr_save=True, path=args_ds.out)
            attacker.run_attack(audio=audio, lengths=length, target=[[toks.index(x) for x in args_ds.target]] * 1,
                                trans=trans, sample_rate=sample_rate,
                                th_batch=th_batch, psd_max_batch=psd_max_batch, masks=masks, masks_freq=masks_freq,
                                maxlen=maxlen, lr_stage2=FLAGS.lr_stage2)


if __name__ == '__main__':
    input_audio_path = './audio_carry/4s/01_English_0070.wav'
    target = "okay google turn off the light"
    main(input_audio_path=input_audio_path, target=target)
