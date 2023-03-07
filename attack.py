# coding=utf-8
# Attack.py -- generate audio adversarial examples
import os
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
import librosa
import struct
import time
from shutil import copyfile

sys.path.append("DeepSpeech")
#
try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置为1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出
import tensorflow as tf

print(tf.test.is_gpu_available())
tf.reset_default_graph()
# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"  # 26  #26个字母

# path = ['./attack_sample/good_time.wav']
path = ['./audio_carry/4s/01_English_0070.wav']
filename = path[0].split('/')[-1]
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
                    default='./deepspeech-0.4.1-checkpoint/model.v0.4.1',
                    help="Path to the ds -ctc checkpoint (ending in model0.4.1)")
args = parser.parse_args()


def convert_mp3(new, lengths):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i + 2])[0] for i in range(0, len(raw.raw_data), 2)])[
            np.newaxis, :lengths[0]]
    return mp3ed


class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        Set up the Attack procedure.
        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """

        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        # zero = np.zeros((batch_size, max_audio_len), dtype=np.float32)
        # self.me = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name="qq_m")
        # self.ve = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name="qq_v")
        # self.te = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name="qq_t")
        # self.deltae = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name="qq_te")
        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.

        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_original')
        self.place_grad_ds = tf.placeholder(tf.float32, shape=(batch_size, max_audio_len))

        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size, 1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -8000, 8000) * self.rescale

        # We set the new input to the audio_model to be the above delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta * mask + original  # mask参数是固定修改的方向，apply—delta是确定修改的量 (1,51072)

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape, stddev=2)  # (1,51072)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)  # 对音频加入一些噪音增强样本的鲁棒性(1,51072)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)
        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])  # 用于保存模型，以后再用就可以直接导入模型进行计算，方便
        saver.restore(sess, restore_path)

        # Choose the loss function we want -- either CTC or CW
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

        # print("start_vars,",tf.global_variables())
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.grad, var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(self.place_grad_ds), var)])
        # self.grad, = tf.gradients(self.loss, [delta])
        # self.train = self.adam_opt(self.delta, 2 * self.grad)
        # self.delta = self.train

        end_vars = tf.global_variables()
        self.global_var = end_vars
        new_vars = [x for x in end_vars if x.name not in start_vars]
        sess.run(tf.variables_initializer(new_vars + [delta]))
        # self.sess.run(tf.initializers.global_variables())

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

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

    def attack(self, audio, lengths, target, finetune=None, itr_save=False, path=None, out_path=None):
        sess = self.sess
        # TODO: each of these assign ops creates a new TF graph
        # sess.run(tf.variables_initializer([self.me, self.ve, self.te, self.deltae]))
        sess.run(tf.variables_initializer([self.delta]))
        # zero = np.zeros((self.batch_size, self.max_audio_len), dtype=np.float32)
        # sess.run(self.me.assign(zero))
        # sess.run(self.ve.assign(zero))
        # sess.run(self.te.assign(zero))
        # sess.run(self.deltae.assign(zero))

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

                if self.mp3:
                    mp3ed = convert_mp3(new, lengths)
                    mp3_out, mp3_logits = sess.run((self.decoded, self.logits), {self.new_input: mp3ed})
                    lst.append((mp3_out, mp3_logits))

                for out, logits in lst:
                    out[0].values
                    res = np.zeros(out[0].dense_shape) + len(toks) - 1
                    for ii in range(len(out[0].values)):
                        x, y = out[0].indices[ii]
                        res[x, y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]
                    print("res is:", "/n".join(res))

                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits, axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(loss - 1) // 320]) for y, loss in zip(res2, lengths)]
                    print("res2:", "/n".join(res2))

            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
            grad_ds = sess.run(self.grad, feed_dict)
            feed_dict_new = {self.place_grad_ds: grad_ds}
            delta, el, ctcloss, loss, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                                       self.ctcloss, self.loss,
                                                                       self.logits, self.new_input,
                                                                       self.train),
                                                                      feed_dict_new)
            plot_loss1.append(ctcloss)
            plot_loss2.append(loss)
            print("loss:", loss)
            print("%.3f" % np.mean(ctcloss), "/t", "/t".join("%.3f" % x for x in ctcloss))  # %.3f保留3位小数，/t表示同行

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
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    np.savetxt('./attack_sample/adv.txt', new_input)

            if itr_save is True:
                name = str(i) + '_' + path[0].split('/')[-1]
                path1 = out_path + '/' + name
                wav.write(path1, 16000,
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


# def main():
def main(out_path):
    """
    Do the Attack here.
    This is all just boilerplate; nothing interesting
    happens in this method.
    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """

    while len(sys.argv) > 1:
        sys.argv.pop()
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []

        if args.out is None:
            assert args.outprefix is not None
        else:
            assert args.outprefix is None
            assert len(args.input) == len(args.out)
        if args.finetune is not None and len(args.finetune):
            assert len(args.input) == len(args.finetune)

        # Load the inputs that we're given
        for i in range(len(args.input)):
            fs, audio = wav.read(args.input[i])  # args.input是一个list{1}的类型
            print(fs, audio, audio.shape)

            np.savetxt('./attack_sample/audio.txt', audio)  # 保留源文件以做比较

            assert fs == 16000
            assert audio.dtype == np.int16
            print('source dB', 20 * np.log10(np.max(np.abs(audio))))  # 89.6297
            audios.append(list(audio))  # 注意：list:1[[audio]]
            lengths.append(len(audio))

            if args.finetune is not None:
                finetune.append(list(wav.read(args.finetune[i])[1]))

        maxlen = max(map(len, audios))  # list:1 51072
        audios = np.array([x + [0] * (maxlen - len(x)) for x in audios])  # np.audios (1,51072)
        finetune = np.array([x + [0] * (maxlen - len(x)) for x in finetune])
        print("lenaudio", len(audios))
        phrase = args.target
        # print(phrase,len(phrase))
        # Set up the Attack class and run it

        attack = Attack(sess, 'CTC', len(phrase), maxlen,  # 14
                        batch_size=1,
                        mp3=args.mp3,
                        learning_rate=args.lr,
                        num_iterations=args.iterations,
                        l2penalty=args.l2penalty,
                        restore_path=args.restore_path)
        deltas = attack.attack(audios, lengths, [[toks.index(x) for x in phrase]] * 1, finetune,
                               itr_save=True, path=args.out, out_path=out_path)
        # len(audios)=1

        # And now save it to the desired output
        if args.mp3:
            convert_mp3(deltas, lengths)
            copyfile("/tmp/saved.mp3", args.out[0])
            print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]] - audios[0][:lengths[0]])))
        else:
            for i in range(len(args.input)):
                if args.out is not None:
                    path = args.out[i]
                else:
                    path = args.outprefix + str(i) + ".wav"
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]] - audios[i][:lengths[i]])))

        np.savetxt('./attack_sample/final_adv.txt', deltas)


if __name__ == '__main__':

    # librosa.griffinlim()
    # assert fs == 8000 is sample require 16000,you can change audio with 8000 or change 8000 to 16000
    # input_audio_path = '/home/guofeng/audio_adversaril/DeepSpeechCTC/test_audio/10s/01_English_0070.wav'
    input_audio_path = './audio_carry/4s/01_English_0070.wav'
    target = "okay google turn off the light"  # because of toks  not contains ',' so ban ','
    out_path = './attack_sample/defense_adv'
    restore_path = './deepspeech-0.4.1-checkpoint/model.v0.4.1'
    itr = True
    main(out_path)
