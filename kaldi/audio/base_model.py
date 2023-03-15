import io
import math
import numpy as np
import os
from six.moves import xrange
# from SpecAugment.tests.spec_augment_test_TF import Mel_main
# from SpecAugment.tests.spec_augment_test_pytorch import Mel_main

import tensorflow as tf


# tf = tf.compat.v1
# tf.disable_v2_behavior()


class array:
    def return_array(self, path):
        with open(path) as f:
            line = f.readline()
            line = line.strip()
            Array = []
            Array = line.split(' ')
            del (Array[len(Array) - 1])
            del (Array[0])
            Array = list(map(float, Array))
            return Array


file1 = '/data/guofeng/Ensembleattck/kaldi/parameter/BL0.txt'
file2 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT0.txt'
file3 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT1.txt'
file4 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT2.txt'
file5 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT3.txt'
file6 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT4.txt'
file7 = '/data/guofeng/Ensembleattck/kaldi/parameter/BT5.txt'
file8 = '/data/guofeng/Ensembleattck/kaldi/parameter/B_C.txt'
file9 = '/data/guofeng/Ensembleattck/kaldi/parameter/B_F.txt'


class Return_value:
    def Seg_L0(self, string):
        with open(string) as f:
            A = []
            line = f.readline()
            for line in f:
                vec = line.strip()
                vecx = vec.split(" ")
                A.append(vecx)
            del (A[len(A) - 1][len(A[len(A) - 1]) - 1])
            for x in range(0, len(A)):
                A[x] = list(map(eval, A[x]))
            matrix = np.matrix(A)
            Matrix = np.transpose(matrix)
        m = []
        n = []
        N = 40
        for i in range(3):
            m.append(Matrix[i * N:(i + 1) * N, :])
        m.append(Matrix[120:220, :])
        for j in range(4):
            n.append(m[j].tolist())
        return n

    def Seg_T0(self, string):
        with open(string) as f:
            A = []
            line = f.readline()
            for line in f:
                vec = line.strip()
                vecx = vec.split(" ")
                A.append(vecx)
            del (A[len(A) - 1][len(A[len(A) - 1]) - 1])
            for x in range(0, len(A)):
                A[x] = list(map(eval, A[x]))
            matrix = np.matrix(A)
            Matrix = np.transpose(matrix)
        Return_T0 = []
        vec = Matrix.tolist()
        Return_T0.append(vec)
        return Return_T0

    def Seg_T1(self, string):
        with open(string) as f:
            A = []
            line = f.readline()
            for line in f:
                vec = line.strip()
                vecx = vec.split(" ")
                A.append(vecx)
            del (A[len(A) - 1][len(A[len(A) - 1]) - 1])
            for x in range(0, len(A)):
                A[x] = list(map(eval, A[x]))
            matrix = np.matrix(A)
            Matrix = np.transpose(matrix)
        m = []
        n = []
        N = 1024
        for i in range(4):
            m.append(Matrix[i * N:(i + 1) * N, :])
        for j in range(4):
            n.append(m[j].tolist())
        return n

    def Seg_all(self, string):
        with open(string) as f:
            A = []
            line = f.readline()
            for line in f:
                vec = line.strip()
                vecx = vec.split(" ")
                A.append(vecx)
            del (A[len(A) - 1][len(A[len(A) - 1]) - 1])
            for x in range(0, len(A)):
                A[x] = list(map(eval, A[x]))
            matrix = np.matrix(A)
            Matrix = np.transpose(matrix)
        O = []
        N = 1024
        for i in range(N):
            temp_O = []
            for j in range(N):
                temp_O.append(0)
            O.append(temp_O)
        m = []
        n = []
        for i in range(3):
            m.append(Matrix[i * N:(i + 1) * N, :])

        for j in range(7):
            if (j % 3 == 0):
                n.append(m[j / 3].tolist())
            else:
                n.append(O)
        return n

    def value(self):
        file0_name = '/data/guofeng/Ensembleattck/kaldi/parameter/T0.txt'
        file1_name = '/data/guofeng/Ensembleattck/kaldi/parameter/C.txt'
        file2_name = '/data/guofeng/Ensembleattck/kaldi/parameter/F_C.txt'
        file3_name = '/data/guofeng/Ensembleattck/kaldi/parameter/L0.txt'

        Return_T0 = self.Seg_T0(file0_name)

        Return_X = self.Seg_T0(file1_name)
        Return_F = self.Seg_T0(file2_name)
        Return_L0 = self.Seg_T0(file3_name)
        return Return_T0, Return_X, Return_F, Return_L0


def read_my_file_format(filename_queue, n_feature):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.]] * n_feature
    example = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')
    return tf.stack(example)


def input_pipeline(filenames, n_feature, batch_size, input_threads=1, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example_list = [read_my_file_format(filename_queue, n_feature)
                    for _ in range(input_threads)]
    example_batch = tf.train.batch(example_list, batch_size=batch_size)
    return example_batch


def l0reformat(examples, ivectors):
    n_frame, m_feature = examples.get_shape().as_list()

    mfcc_first = tf.slice(examples, [0, 0], [1, m_feature])  # mfcc of the first frame
    mfcc_last = tf.slice(examples, [n_frame - 1, 0], [1, m_feature])  # mfcc of the last frame

    examples_pad = tf.concat([tf.tile(mfcc_first, [17, 1]), examples, tf.tile(mfcc_last, [12, 1])], 0)
    print(examples_pad.shape)

    n_batch = 10  # number of batches

    h0 = tf.zeros([0, 50 + 27, 220])

    for i_batch in xrange(0, n_batch):
        print('Batch:{0}'.format(i_batch))

        istart = 50 * i_batch
        iend = min(n_frame, 50 * (i_batch + 1)) + 27
        ivector_frame = i_batch * 5 + 3 - 1

        print(istart)
        print(iend)
        print(ivector_frame)

        h0_temp = tf.zeros([0, 220])
        for i_frame in xrange(istart, iend):
            print(i_frame)
            mfcc0 = tf.slice(examples_pad, [i_frame, 0], [1, m_feature])
            mfcc1 = tf.slice(examples_pad, [i_frame + 1, 0], [1, m_feature])
            mfcc2 = tf.slice(examples_pad, [i_frame + 2, 0], [1, m_feature])
            ivector_temp = tf.slice(ivectors, [ivector_frame, 0], [1, 100])
            h0_temp = tf.concat([h0_temp, tf.concat([mfcc0, mfcc1, mfcc2, ivector_temp], 1)], 0)

        for i_frame in xrange(iend, 50 * (i_batch + 1) + 27):
            print(i_frame)
            zero_padding = tf.zeros([1, 220])
            h0_temp = tf.concat([h0_temp, zero_padding], 0)

        h0 = tf.concat([h0, tf.expand_dims(h0_temp, 0)], 0)

    print(h0.shape)

    return h0


def inversel0(advIn):
    n_batch, n_frame, m_feature = advIn.get_shape().as_list()
    mfcc = tf.zeros([1, 0, 40])

    for i in xrange(0, n_batch):
        mfcc_temp = tf.slice(advIn, [i, 16, 40], [1, 50, 40])
        mfcc = tf.concat([mfcc, mfcc_temp], 1)

    return tf.squeeze(mfcc)


def window_povel(wav):
    #
    frame_length = tf.shape(wav)[0]
    frame_length = tf.cast(frame_length, dtype=tf.float32)
    #  frame_pad=pow(2,math.ceil(math.log(frame_length,2)))
    frame_pad = 256
    dc_offset = tf.reduce_sum(wav) / frame_length
    wav = wav - dc_offset  # right

    ses1 = tf.Session()
    frame_length_out = np.int32(ses1.run(frame_length))
    ses1.close()
    print(frame_length_out)

    # pre_emphasize
    preemph_coeff = 0.97
    wav_set = wav[0:frame_length_out - 1]
    t_ones = wav[0]
    t_ones = t_ones * (1 - preemph_coeff) / preemph_coeff
    t_ones = tf.reshape(t_ones, shape=[1, 1])
    wav_set = tf.reshape(wav_set, shape=[1, frame_length_out - 1])
    wav_set = tf.concat([t_ones, wav_set], axis=1)
    wav = wav - wav_set * preemph_coeff
    # window function
    M_2PI = 6.283185307179586476925286766559005
    a = M_2PI / (frame_length_out - 1)  # r

    win_p = np.zeros((1, frame_length_out), dtype=np.float32)
    for i in range(0, frame_length_out):
        win_p[0, i] = pow(0.5 - 0.5 * math.cos(a * (i + 1)), 0.85)  # r
    wav = tf.multiply(wav, win_p)  # r--

    t_zeros = tf.zeros(shape=[1, frame_pad - frame_length_out], dtype=tf.float32)
    wav = tf.concat([wav, t_zeros], axis=1)

    return wav


def dct(M_m):
    num_filt = 40
    sq1 = np.linspace(0, 39, 40)
    sq2 = np.linspace(0.5, 39.5, 40)

    sq11 = np.zeros(shape=(40, 1), dtype="float32")
    sq22 = np.zeros(shape=(1, 40), dtype="float32")
    sq11[:, 0] = sq1
    sq22[0, :] = sq2
    sq_dct = 1 / math.sqrt(num_filt / 2) * np.cos(np.dot(sq11, sq22) * math.pi / num_filt)
    sq_dct[0, :] = sq_dct[0, :] * math.sqrt(2) / 2
    mfcc = tf.matmul(sq_dct, M_m)
    return mfcc


def coe_lift(mfcc, num_segments_t):
    lift = 22
    # frame_length=tf.shape(wav)[0]
    coeff = np.ones(shape=(40, np.int(num_segments_t)), dtype="float32")
    for i in range(0, 40):
        coeff[i, :] = coeff[i, :] * (1 + 0.5 * 22 * math.sin(math.pi * i / 22))
    mfcc = mfcc * coeff
    return mfcc


def acoustic_model(wav, num_segments_t):
    num_ceps_coeffs = 40
    FS = 8000
    num_filt = 40
    seg_size = FS * 0.025
    hop_size = FS * 0.01
    # wav=tf.convert_to_tensor(sig,dtype=tf.float32)
    print("wav:", wav)
    print(num_segments_t)
    P_out = tf.zeros(shape=[1, 256], dtype=tf.float32)
    for i in range(0, np.int32(num_segments_t)):
        # idx=np.linspace(0,seg_size,seg_size)+hop_size*i
        idx_s = np.int32(hop_size * i)
        idx_e = np.int32(seg_size + hop_size * i)
        wav_i = wav[idx_s:idx_e]  # right
        wav_i = window_povel(wav_i)

        wav_i_f = tf.cast(wav_i, dtype=tf.complex64)
        wav_i_f = tf.fft(wav_i_f)  #
        wav_out = tf.abs(wav_i_f)
        wav_out = tf.multiply(wav_out, wav_out)
        if i == 0:
            P_out = wav_out
        else:
            P_out = tf.concat([P_out, wav_out], axis=0)  # r-

    P_out_half = P_out[:, 0:129]
    Mel_filter = np.loadtxt(open("/data/guofeng/Ensembleattck/kaldi/zy_tensor/Mel_filter.csv", "rb"), delimiter=",",
                            dtype="float32")
    print(" Mel_filter :", Mel_filter.shape)

    M_m = tf.matmul(Mel_filter, tf.transpose(P_out_half))
    print("Mel filter:", M_m.shape)  # (40,425)
    M_m = M_m + 0.000000001
    print("Mel filter:", M_m.shape)
    # +++++++++++++++++++++++++++++++++++++++++++
    Mel_out = M_m
    # st = tf.Session()
    # Mel_out = st.run(Mel)
    # st.close()
    # spc_mel = Mel_main(Mel_out,num_segments_t)
    # print("spc_mel :",spc_mel)
    # M_m = spc_mel
    # +++++++++++++++++++++++++++++++++++++++++++
    M_m = tf.log(M_m)  # r--
    mfcc = dct(M_m)  # r---
    mfcc = coe_lift(mfcc, num_segments_t)

    return mfcc, Mel_out


def tdnn(dnnInput, n_frame):
    # TODO yuxuan: implement the dnn network
    num_line = 224
    A = Return_value()
    B = A.value()
    kernel_t0 = B[0]

    kernel_c = B[1]
    kernel_f = B[2]
    kernel_l0 = B[3]

    a = array()
    BL0 = a.return_array(file1)
    BT0 = a.return_array(file2)
    BT1 = a.return_array(file3)
    BT2 = a.return_array(file4)
    BT3 = a.return_array(file5)
    BT4 = a.return_array(file6)
    BT5 = a.return_array(file7)
    BC = a.return_array(file8)
    BFC = a.return_array(file9)
    # matrix = np.loadtxt(open('xuan_1.csv',"rb"),delimiter = ",", dtype = "float32")
    # l0=tf.reshape(matrix,(1,n_frame+27,220))
    input = tf.reshape(dnnInput, (10, n_frame + 27, 220))
    input11 = input[0, :, :]
    input12 = input[1, :, :]
    input13 = input[2, :, :]
    input14 = input[3, :, :]
    input15 = input[4, :, :]
    input16 = input[5, :, :]
    input17 = input[6, :, :]
    input18 = input[7, :, :]
    input19 = input[8, :, :]
    input110 = input[9, :, :]
    input1 = tf.reshape(input11, (1, 77, 220))
    input2 = tf.reshape(input12, (1, 77, 220))
    input3 = tf.reshape(input13, (1, 77, 220))
    input4 = tf.reshape(input14, (1, 77, 220))
    input5 = tf.reshape(input15, (1, 77, 220))
    input6 = tf.reshape(input16, (1, 77, 220))
    input7 = tf.reshape(input17, (1, 77, 220))
    input8 = tf.reshape(input18, (1, 77, 220))
    input9 = tf.reshape(input19, (1, 77, 220))
    input10 = tf.reshape(input110, (1, 77, 220))
    final_relu_p1 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p2 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p3 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p4 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p5 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p6 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p7 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p8 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p9 = tf.zeros([1, 50, 8629], dtype=tf.float32)
    final_relu_p10 = tf.zeros([1, 50, 8629], dtype=tf.float32)

    # T1_affine_input = np.load('kaldi_t1_affine_input.np.npy')
    T1_input_p = np.loadtxt(open('/data/guofeng/Ensembleattck/kaldi/parameter/T1.csv', "rb"), delimiter=",", dtype="float32")
    T1_input = np.reshape(T1_input_p, (1024, 4096))
    T1 = T1_input.T
    T1_1 = T1[0:1024, :]
    T1_2 = T1[1024:2048, :]
    T1_3 = T1[2048:3072, :]
    T1_4 = T1[3072:4096, :]
    T1 = np.stack((T1_1, T1_2, T1_3, T1_4), axis=0)
    T2_input_p = np.loadtxt(open('/data/guofeng/Ensembleattck/kaldi/parameter/T2.csv', "rb"), delimiter=",", dtype="float32")
    T2_input = np.reshape(T2_input_p, (1024, 3072))
    T2 = T2_input.T
    T2_0 = T2[0:1024, :]
    T2_1 = np.zeros((1024, 1024), dtype=np.float32)
    T2_2 = np.zeros((1024, 1024), dtype=np.float32)
    T2_3 = T2[1024:2048, :]
    T2_4 = np.zeros((1024, 1024), dtype=np.float32)
    T2_5 = np.zeros((1024, 1024), dtype=np.float32)
    T2_6 = T2[2048:3072, :]
    T2 = np.stack((T2_0, T2_1, T2_2, T2_3, T2_4, T2_5, T2_6), axis=0)
    T3_input_p = np.loadtxt(open('/data/guofeng/Ensembleattck/kaldi/parameter/T3.csv', "rb"), delimiter=",", dtype="float32")
    T3_input = np.reshape(T3_input_p, (1024, 3072))
    T3 = T3_input.T
    T3_0 = T3[0:1024, :]
    T3_1 = np.zeros((1024, 1024), dtype=np.float32)
    T3_2 = np.zeros((1024, 1024), dtype=np.float32)
    T3_3 = T3[1024:2048, :]
    T3_4 = np.zeros((1024, 1024), dtype=np.float32)
    T3_5 = np.zeros((1024, 1024), dtype=np.float32)
    T3_6 = T3[2048:3072, :]
    T3 = np.stack((T3_0, T3_1, T3_2, T3_3, T3_4, T3_5, T3_6), axis=0)
    T4_input_p = np.loadtxt(open('/data/guofeng/Ensembleattck/kaldi/parameter/T4.csv', "rb"), delimiter=",", dtype="float32")
    T4_input = np.reshape(T4_input_p, (1024, 3072))
    T4 = T4_input.T
    T4_0 = T4[0:1024, :]
    T4_1 = np.zeros((1024, 1024), dtype=np.float32)
    T4_2 = np.zeros((1024, 1024), dtype=np.float32)
    T4_3 = T4[1024:2048, :]
    T4_4 = np.zeros((1024, 1024), dtype=np.float32)
    T4_5 = np.zeros((1024, 1024), dtype=np.float32)
    T4_6 = T4[2048:3072, :]
    T4 = np.stack((T4_0, T4_1, T4_2, T4_3, T4_4, T4_5, T4_6), axis=0)
    T5_input_p = np.loadtxt(open('/data/guofeng/Ensembleattck/kaldi/parameter/T5.csv', "rb"), delimiter=",", dtype="float32")
    T5_input = np.reshape(T5_input_p, (1024, 3072))
    T5 = T5_input.T
    T5_0 = T5[0:1024, :]
    T5_1 = np.zeros((1024, 1024), dtype=np.float32)
    T5_2 = np.zeros((1024, 1024), dtype=np.float32)
    T5_3 = T5[1024:2048, :]
    T5_4 = np.zeros((1024, 1024), dtype=np.float32)
    T5_5 = np.zeros((1024, 1024), dtype=np.float32)
    T5_6 = T5[2048:3072, :]
    T5 = np.stack((T5_0, T5_1, T5_2, T5_3, T5_4, T5_5, T5_6), axis=0)

    for i in range(1, 11):
        l0_affine = tf.nn.conv1d(locals()['input' + str(i)], kernel_l0, 1, 'SAME')
        t0_affine = tf.nn.conv1d(tf.nn.bias_add(l0_affine, BL0), kernel_t0, 1, 'SAME')
        t0_relu = tf.nn.relu(tf.nn.bias_add(t0_affine, BT0))
        t0_renorm_input = tf.reshape(t0_relu, (n_frame + 27, 1024))
        rmst0_p = tf.sqrt(tf.reduce_mean(tf.square(t0_renorm_input), 1))
        rmst0 = tf.reshape(rmst0_p, (n_frame + 27, 1))
        onest0 = tf.ones([1, 1024])
        rms_t0 = tf.multiply(rmst0, onest0)
        t0_renorm = tf.math.divide(t0_renorm_input, rms_t0)
        t0_renorm3 = tf.reshape(t0_renorm, (1, 1, n_frame + 27, 1024))
        # end T0 layer

        # start T1 layer
        print('T1 Layer')
        # %%%%%%%%%%%%%%%%%%%tensor-process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t1_input1 = t0_renorm3[0, 0, 0:n_frame + 24, :]
        t1_input2 = t0_renorm3[0, 0, 1:n_frame + 25, :]
        t1_input3 = t0_renorm3[0, 0, 2:n_frame + 26, :]
        t1_input4 = t0_renorm3[0, 0, 3:n_frame + 27, :]
        t1_input_final = tf.concat([t1_input1, t1_input2, t1_input3, t1_input4], axis=1)
        t1_zeros = tf.zeros(shape=[num_line - (n_frame + 24), 4096], dtype=tf.float32)
        t1_input_final = tf.concat([t1_input_final, t1_zeros], axis=0)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        t1_affine_p = tf.nn.conv1d(tf.reshape(t1_input_final, (4, num_line, 1024)),
                                   tf.convert_to_tensor(T1, np.float32), 1, 'SAME')
        t1_affine_f = tf.reshape(t1_affine_p, [num_line, 4096])
        t1_affine_f = t1_affine_f[0:n_frame + 24, 1024:2048]

        t1_relu_p = tf.nn.bias_add(t1_affine_f, BT1)
        t1_relu = tf.nn.relu(t1_relu_p)

        t1_renorm_input = tf.reshape(t1_relu, (n_frame + 24, 1024))
        rmst1_p = tf.sqrt(tf.reduce_mean(tf.square(t1_renorm_input), 1))
        rmst1 = tf.reshape(rmst1_p, (n_frame + 24, 1))
        onest1 = tf.ones([1, 1024])
        rms_t1 = tf.multiply(rmst1, onest1)
        t1_renorm = tf.math.divide(t1_renorm_input, rms_t1)
        t1_renorm3 = tf.reshape(t1_renorm, (1, 1, n_frame + 24, 1024))
        # end T1

        # start T2
        print('T2 Layer')
        # %%%%%%%%%%%%%%%%%%%tensor-process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t2_input1 = t1_renorm3[0, 0, 0:n_frame + 18, :]
        t2_input_zeros = tf.zeros(shape=[n_frame + 18, 1024], dtype=tf.float32)
        t2_input2 = t1_renorm3[0, 0, 3:n_frame + 21, :]
        t2_input3 = t1_renorm3[0, 0, 6:n_frame + 24, :]
        t2_input_final = tf.concat(
            [t2_input1, t2_input_zeros, t2_input_zeros, t2_input2, t2_input_zeros, t2_input_zeros, t2_input3], axis=1)
        t2_zeros = tf.zeros(shape=[num_line - (n_frame + 18), 7168], dtype=tf.float32)
        t2_input_final = tf.concat([t2_input_final, t2_zeros], axis=0)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        t2_affine_p = tf.nn.conv1d(tf.reshape(t2_input_final, (7, num_line, 1024)),
                                   tf.convert_to_tensor(T2, np.float32), 1, 'SAME')
        t2_affine_f = tf.reshape(t2_affine_p, [num_line, 7168])
        t2_affine_f = t2_affine_f[0:n_frame + 18, 3072:4096]

        t2_relu_p = tf.nn.bias_add(t2_affine_f, BT2)
        t2_relu = tf.nn.relu(t2_relu_p)

        t2_renorm_input = tf.reshape(t2_relu, (n_frame + 18, 1024))
        rmst2_p = tf.sqrt(tf.reduce_mean(tf.square(t2_renorm_input), 1))
        rmst2 = tf.reshape(rmst2_p, (n_frame + 18, 1))
        onest2 = tf.ones([1, 1024])
        rms_t2 = tf.multiply(rmst2, onest2)
        t2_renorm = tf.math.divide(t2_renorm_input, rms_t2)
        t2_renorm3 = tf.reshape(t2_renorm, (1, 1, n_frame + 18, 1024))
        # end T2 layer
        # start T3 layer
        print('T3 Layer')
        # %%%%%%%%%%%%%%%%%%%tensor-process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t3_input1 = t2_renorm3[0, 0, 0:n_frame + 12, :]
        t3_input_zeros = tf.zeros(shape=[n_frame + 12, 1024], dtype=tf.float32)
        t3_input2 = t2_renorm3[0, 0, 3:n_frame + 15, :]
        t3_input3 = t2_renorm3[0, 0, 6:n_frame + 18, :]
        t3_input_final = tf.concat(
            [t3_input1, t3_input_zeros, t3_input_zeros, t3_input2, t3_input_zeros, t3_input_zeros, t3_input3], axis=1)
        t3_zeros = tf.zeros(shape=[num_line - (n_frame + 12), 7168], dtype=tf.float32)
        t3_input_final = tf.concat([t3_input_final, t3_zeros], axis=0)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        t3_affine_p = tf.nn.conv1d(tf.reshape(t3_input_final, (7, num_line, 1024)),
                                   tf.convert_to_tensor(T3, np.float32), 1, 'SAME')
        t3_affine_f = tf.reshape(t3_affine_p, [num_line, 7168])
        t3_affine_f = t3_affine_f[0:n_frame + 12, 3072:4096]

        t3_relu_p = tf.nn.bias_add(t3_affine_f, BT3)
        t3_relu = tf.nn.relu(t3_relu_p)

        t3_renorm_input = tf.reshape(t3_relu, (n_frame + 12, 1024))
        rmst3_p = tf.sqrt(tf.reduce_mean(tf.square(t3_renorm_input), 1))
        rmst3 = tf.reshape(rmst3_p, (n_frame + 12, 1))
        onest3 = tf.ones([1, 1024])
        rms_t3 = tf.multiply(rmst3, onest3)
        t3_renorm = tf.math.divide(t3_renorm_input, rms_t3)
        t3_renorm3 = tf.reshape(t3_renorm, (1, 1, n_frame + 12, 1024))
        # end T3 layer

        # start T4 layer
        # ses41=tf.Session()
        # t4input=ses41.run(t3_renorm3)
        # np.save("xuan/t3renorm.np",t4input)
        # t4_input=t4input[0,0,:,:]
        # ses41.close()
        print('T4 Layer')
        # %%%%%%%%%%%%%%%%%%%tensor-process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t4_input1 = t3_renorm3[0, 0, 0:n_frame + 6, :]
        t4_input_zeros = tf.zeros(shape=[n_frame + 6, 1024], dtype=tf.float32)
        t4_input2 = t3_renorm3[0, 0, 3:n_frame + 9, :]
        t4_input3 = t3_renorm3[0, 0, 6:n_frame + 12, :]
        t4_input_final = tf.concat(
            [t4_input1, t4_input_zeros, t4_input_zeros, t4_input2, t4_input_zeros, t4_input_zeros, t4_input3], axis=1)
        t4_zeros = tf.zeros(shape=[num_line - (n_frame + 6), 7168], dtype=tf.float32)
        t4_input_final = tf.concat([t4_input_final, t4_zeros], axis=0)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        t4_affine_p = tf.nn.conv1d(tf.reshape(t4_input_final, (7, num_line, 1024)),
                                   tf.convert_to_tensor(T4, np.float32), 1, 'SAME')
        t4_affine_f = tf.reshape(t4_affine_p, [num_line, 7168])
        t4_affine_f = t4_affine_f[0:n_frame + 6, 3072:4096]

        t4_relu_p = tf.nn.bias_add(t4_affine_f, BT4)
        t4_relu = tf.nn.relu(t4_relu_p)

        t4_renorm_input = tf.reshape(t4_relu, (n_frame + 6, 1024))
        rmst4_p = tf.sqrt(tf.reduce_mean(tf.square(t4_renorm_input), 1))
        rmst4 = tf.reshape(rmst4_p, (n_frame + 6, 1))
        onest4 = tf.ones([1, 1024])
        rms_t4 = tf.multiply(rmst4, onest4)
        t4_renorm = tf.math.divide(t4_renorm_input, rms_t4)
        t4_renorm3 = tf.reshape(t4_renorm, (1, 1, n_frame + 6, 1024))
        # end T4 layer
        # start T5 layer
        print('T5 Layer')
        # %%%%%%%%%%%%%%%%%%%tensor-process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t5_input1 = t4_renorm3[0, 0, 0:n_frame, :]
        t5_input_zeros = tf.zeros(shape=[n_frame, 1024], dtype=tf.float32)
        t5_input2 = t4_renorm3[0, 0, 3:n_frame + 3, :]
        t5_input3 = t4_renorm3[0, 0, 6:n_frame + 6, :]
        t5_input_final = tf.concat(
            [t5_input1, t5_input_zeros, t5_input_zeros, t5_input2, t5_input_zeros, t5_input_zeros, t5_input3], axis=1)
        t5_zeros = tf.zeros(shape=[num_line - n_frame, 7168], dtype=tf.float32)
        t5_input_final = tf.concat([t5_input_final, t5_zeros], axis=0)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        t5_affine_p = tf.nn.conv1d(tf.reshape(t5_input_final, (7, num_line, 1024)),
                                   tf.convert_to_tensor(T5, np.float32), 1, 'SAME')
        t5_affine_f = tf.reshape(t5_affine_p, [num_line, 7168])
        t5_affine_f = t5_affine_f[0:n_frame, 3072:4096]

        t5_relu_p = tf.nn.bias_add(t5_affine_f, BT5)
        t5_relu = tf.nn.relu(t5_relu_p)

        t5_renorm_input = tf.reshape(t5_relu, (n_frame, 1024))
        rmst5_p = tf.sqrt(tf.reduce_mean(tf.square(t5_renorm_input), 1))
        rmst5 = tf.reshape(rmst5_p, (n_frame, 1))
        onest5 = tf.ones([1, 1024])
        rms_t5 = tf.multiply(rmst5, onest5)
        t5_renorm = tf.math.divide(t5_renorm_input, rms_t5)
        t5_renorm3 = tf.reshape(t5_renorm, (1, 1, n_frame, 1024))
        # end T5 layer

        # start chain_layer
        print('Chain Layer')
        chain_affine = tf.nn.conv1d(tf.reshape((t5_renorm3), (1, n_frame, 1024)), kernel_c, 1, 'SAME')

        chain_relu_p = tf.nn.bias_add(chain_affine, BC)
        chain_relu = tf.nn.relu(chain_relu_p)

        chain_renorm_input = tf.reshape(chain_relu, (n_frame, 1024))
        rmschain_p = tf.sqrt(tf.reduce_mean(tf.square(chain_renorm_input), 1))
        rmschain = tf.reshape(rmschain_p, (n_frame, 1))
        oneschain = tf.ones([1, 1024])
        rms_chain = tf.multiply(rmschain, oneschain)
        chain_renorm = tf.math.divide(chain_renorm_input, 2 * rms_chain)
        chain_renorm3 = tf.reshape(chain_renorm, (1, 1, n_frame, 1024))

        # end chain layer
        # start final layer
        # ses73=tf.Session()
        # chain_output=ses73.run(chain_renorm3)
        # np.save("xuan/chainrenorm.np",chain_output)
        # ses73.close()
        print('Final Layer')
        final_affine = tf.nn.conv1d(tf.reshape(chain_renorm3, (1, n_frame, 1024)), kernel_f, 1, 'SAME')

        if i == 1:
            final_relu_p1 = tf.nn.bias_add(final_affine, BFC)
        if i == 2:
            final_relu_p2 = tf.nn.bias_add(final_affine, BFC)
        if i == 3:
            final_relu_p3 = tf.nn.bias_add(final_affine, BFC)
        if i == 4:
            final_relu_p4 = tf.nn.bias_add(final_affine, BFC)
        if i == 5:
            final_relu_p5 = tf.nn.bias_add(final_affine, BFC)
        if i == 6:
            final_relu_p6 = tf.nn.bias_add(final_affine, BFC)
        if i == 7:
            final_relu_p7 = tf.nn.bias_add(final_affine, BFC)
        if i == 8:
            final_relu_p8 = tf.nn.bias_add(final_affine, BFC)
        if i == 9:
            final_relu_p9 = tf.nn.bias_add(final_affine, BFC)
        if i == 10:
            final_relu_p10 = tf.nn.bias_add(final_affine, BFC)

    final_relu_p = tf.concat(
        [final_relu_p1, final_relu_p2, final_relu_p3, final_relu_p4, final_relu_p5, final_relu_p6, final_relu_p7,
         final_relu_p8, final_relu_p9, final_relu_p10], axis=0)
    final_relu_pp = tf.reshape(final_relu_p, (1, 500, 8629))
    return final_relu_pp


def google_api_phone(iteration_ref, noise_ref, loss_ref, iteration_time, music_num, speech=None):
    # Instantiates a client
    client = speech.SpeechClient()
    path = "csv_to_wav/yuxuan44/music%s_sample_iteration%s_noise%s_loss%s_iterationnumber%s_time.wav" % (
        music_num, iteration_ref, noise_ref, loss_ref, iteration_time)

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        path)

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.types.RecognitionAudio(content=content)

    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US',
        sample_rate_hertz=8000,
        model='phone_call'
    )

    # Detects speech in the audio file
    global response
    response = client.recognize(config, audio)
    print(response.results)

    f = open('/udrive/student/yuxuan2015/AfterDevilTXT/phone_googlerecordairplane_music%s.txt' % (music_num), 'a')

    f.write(str(response.results))
    f.write('\n The file is %s' % (path))

    f.close()
