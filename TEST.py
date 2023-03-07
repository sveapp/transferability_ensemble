# coding=utf-8
import argparse
import sys

import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from matplotlib import pyplot  as plt

sys.path.append("DeepSpeech")
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

toks = " abcdefghijklmnopqrstuvwxyz'-"  # 26  #26个字母


class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float(0.07), restore_path=None):
        """
        Set up the attack procedure.
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

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                         name='qq_delta')  # tf.Variable 'qq_delta:0' shape=(1, 51072) dtype=float32_ref>
        self.delta1 = delta1 = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                           name='qq_delta1')

        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               # <tf.Variable 'qq_original:0' shape=(1, 51072) dtype=float32_ref>
                                               name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32),
                                             name='qq_lengths')  # <tf.Variable 'qq_lengths:0' shape=(1,) dtype=int32_ref>
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size, 1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000) * self.rescale

        # We set the new input to the model to be the above delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta + original  # mask参数是固定修改的方向，apply—delta是确定修改的量 (1,51072)

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape, stddev=2)  # (1,51072)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)  # 对音频加入一些噪音增强样本的鲁棒性(1,51072)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in,
                                          lengths)  # 获取已经加了deta和noise的audio的mfcc特征
        # Tensor("raw_logits:0", shape=(159, 1, 29), dtype=float32)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])  # 用于保存模型，以后再用就可以直接导入模型进行计算，方便
        saver.restore(sess, restore_path)

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase,
                                               self.target_phrase_lengths)  # sparsetensoir:shape(?,?)
            # sparsetenspr是一个稀疏矩阵
            # a = tf.SparseTensor(indice=[[0, 0], [1, 1]], values=[8, 88], dense_shape=(3, 4))
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),  # ctcloss = {Tensor} Tensor("CTCLoss:0",
                                     inputs=logits, sequence_length=lengths)  # shape=(1,), dtype=float32)
            # tf.cast()函数的作用是执行 tensorflow2
            # 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            # if not np.isinf(l2penalty):
            loss =0.03 * tf.reduce_mean((self.new_input - self.original), axis=1) +  ctcloss
            # loss = 0.03 * ctcloss
            # loss = ctcloss
            self.expanded_loss = tf.constant(0)

        elif loss_fn == "CW":
            raise NotImplemented(
                "The current version of this project does not include the CW loss function implementation.")
        else:
            raise

        self.loss = loss
        self.ctcloss = ctcloss

        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)  # 使用adam优化算法作为梯度下降优化器

        grad, var = optimizer.compute_gradients(self.loss, [delta])[0]
        # grad1, var1 = optimizer.compute_gradients(self.ctcloss, [delta1])[0]
        # 计算梯度 (1,51072)var = <tf.Variable 'qq_delta:0' shape=(1, 51072) >
        print("grad:", grad, "\n", "var:", var)

        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])  # 将计算出的梯度应用到变量上，于使用计算得到的梯度来更新对应的variable,
        # self.train = optimizer.apply_gradients([(tf.sign(grad1), var1)])
        # 这里将tf.sign(grad)更新到var上
        # self.kk=optimizer.minimize()
        end_vars = tf.global_variables()  #
        print("tf.global_varibles:", end_vars)

        new_vars = [x for x in end_vars if x.name not in start_vars]

        sess.run(tf.variables_initializer(new_vars + [delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, audio, lengths, target,
               finetune=None):  # target = {list: 1} [[20, 8, 9, 19, 0, 9, 19, 0, 1, 0, 20, 5, 19, 20]]
        sess = self.sess
        sess.run(tf.variables_initializer([self.delta]))  # np.date(1,51072) (0...0.0.0..)
        sess.run(tf.variables_initializer([self.delta1]))

        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.target_phrase_lengths.assign(
            np.array([len(x) for x in target])))  # target:list(1):[[20, 8, 9, 19, 0, 9, 19, 0, 1, 0, 20, 5, 19, 20]]
        sess.run(self.target_phrase.assign(np.array(
            [list(t) + [0] * (self.phrase_length - len(t)) for t in target])))  # t = {list: 14}
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size, 1))))
        final_deltas = [None] * self.batch_size  # batch_size=1 final_deltas = {list: 1} :None
        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune - audio))

        plot_loss1 = []
        plot_loss2 = []

        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            # if i % 10 == 0:
            new, delta, delta1, r_out, r_logits = sess.run(
                (self.new_input, self.delta, self.delta1, self.decoded, self.logits))
            lst = [(r_out, r_logits)]
            for out, logits in lst:
                out[0].values  # logits=r_logits(159,1,29) out=lst=r_out
                res = np.zeros(out[0].dense_shape) + len(toks) - 1  # res = {ndarray: (1, 43)} [28..[28]]
                for ii in range(len(out[0].values)):
                    x, y = out[0].indices[ii]
                    res[x, y] = out[0].values[ii]
                # Here we print the strings that are recognized.
                res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]
                print("res is:", "\n".join(res))
                # And here we print the argmax of the alignment.
                res2 = np.argmax(logits, axis=2).T
                res2 = ["".join(toks[int(x)] for x in y[:(l - 1) // 320]) for y, l in zip(res2, lengths)]
                print("res2:", "\n".join(res2))  # join()：连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串

            feed_dict = {}
            d, d1, el, cl, l, logits, new_input, _ = sess.run((self.delta, self.delta1, self.expanded_loss,
                                                               self.ctcloss, self.loss,
                                                               self.logits, self.new_input,
                                                               self.train),
                                                              feed_dict)
            plot_loss1.append(cl)
            plot_loss2.append(l)
            print("%.3f" % np.mean(cl), "\t", "\t".join("%.3f" % x for x in cl))  # %.3f保留3位小数，\t表示同行

            np.argmax(logits, axis=2).T
            for ii in range(self.batch_size):
                rescale = sess.run(self.rescale)
                if rescale[ii] * 2000 > np.max(np.abs(d)):
                    print("It's way over", np.max(np.abs(d[ii])) / 2000.0)
                rescale[ii] = np.max(np.abs(d[ii])) / 2000.0  # 此处d代替了delta，rescale
                rescale[ii] *= .8
                final_deltas[ii] = new_input[ii]
            print("Worked i=%d ctcloss=%f bound=%f" % (ii, cl[ii], 2000 * rescale[ii][0]))
            print('delta:', np.max(np.abs(new_input[ii] - audio[ii])))
            sess.run(self.rescale.assign(rescale))

        plt.subplot(1, 2, 1)
        # plt.xlabel("epoch")
        # plt.ylabel("ctcloss")

        plt.plot(plot_loss1, label="$ctcloss$")
        plt.title("ctcloss")
        plt.legend()

        plt.subplot(1, 2, 2)
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        plt.plot(plot_loss2, label="$l$")
        plt.title("loss")
        plt.legend()
        plt.show()

        return final_deltas


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")
    parser.add_argument('--out', type=str, nargs='+',
                        required=False,
                        help="Path for the adversarial example(s)")
    parser.add_argument('--outprefix', type=str,
                        required=False,
                        help="Prefix of path for adversarial examples")
    parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
    parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate for optimization")
    parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
    parser.add_argument('--mp3', action="store_const", const=True,
                        required=False,
                        help="Generate MP3 compression resistant adversarial examples")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()

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

            np.savetxt('audio.txt', audio)  # 保留源文件以做比较

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

        phrase = args.target

        # Set up the attack class and run it
        attack = Attack(sess, 'CTC', len(phrase), maxlen,  # 14
                        batch_size=len(audios),
                        mp3=args.mp3,
                        learning_rate=args.lr,
                        num_iterations=args.iterations,
                        l2penalty=args.l2penalty,
                        restore_path=args.restore_path)
        deltas = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]] * len(audios),
                               finetune)  # len(audios)=1

        if args.mp3:
            max(1, 3)
        else:
            for i in range(len(args.input)):
                if args.out is not None:
                    path = args.out[i]
                else:
                    path = args.outprefix + str(i) + ".wav"
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]] - audios[i][:lengths[i]])))

        np.savetxt('final_adv.txt', deltas)


if __name__ == '__main__':
    main()
