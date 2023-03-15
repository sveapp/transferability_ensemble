# coding=utf-8
# from urllib import request, parse
import scipy.io.wavfile as wav
import json
import os
import urllib2 as request
import urllib as parse
def get_token():
    API_Key = "Lyl2c75iyRgv6fCHPhd56V4X"  # 官网获取的API_Key
    Secret_Key = "YfifTfZyNDoNsPAWsCGXkqd7s0XuFGkr"  # 为官网获取的Secret_Key
    # 拼接得到Url
    Url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + API_Key + "&client_secret=" + Secret_Key
    try:
        resp = request.urlopen(Url)
        result = json.loads(resp.read().decode('utf-8'))
        # 打印access_token
        # print("access_token:", result['access_token'])
        return result['access_token']
    except request.URLError as err:
        # print('token http response http code : ' + str(err.code))
        print('token http response http code : ' + str(err))


def baidu_recog(path):
    # print("baidu recognition start working")
    filename = path.split('/')[-1]
    # print(filename)
    token = get_token()
    params = {'cuid': "sdf454bfg",  # 用户唯一标识，用来区分用户，长度为60字符以内。
              'token': token,  # 我们获取到的 Access Token
              'dev_pid': 1737}  # 1537为普通话输入法模型，1737为英语模型
    params_query = parse.urlencode(params)

    Url = 'http://vop.baidu.com/server_api' + "?" + params_query

    rate, _ = wav.read(path)
    # print("rate is :", rate)
    if rate == 16000:
        with open(path, 'rb') as speech_file:
            speech_data = speech_file.read()
        length = len(speech_data)
        if length == 0:
            print('file 01.wav length read 0 bytes')
        headers = {
            'Content-Type': 'audio/wav; rate=8000',  # 采样率和文件格式
            'Content-Length': length
        }
        req = request.Request(Url, speech_data, headers)
        res_f = request.urlopen(req)
        result = json.loads(res_f.read().decode('utf-8'))
        if 'result' in result:
            res = result['result'][0]
        else:
            res = result['err_no']
            # print(result)

        # print(result)
        # print("识别结果:", result['result'][0])  # result['result'][0] is string
    else:
        raise RuntimeError("rate must is 16000")
    return res, filename


def getfile_path(filepath):
    l_dir = os.listdir(filepath)
    filepaths = []
    for one in l_dir:
        full_path = os.path.join('%s/%s' % (filepath, one))  # 构造路径
        filepaths.append(full_path)
    return filepaths


if __name__ == '__main__':
    # path = "/home/guofeng/kaldi/egs/aspire/s5/audio_wav/d/8k.84-121123-0006.wav"
    # path = "/data/guofeng/Speech_Recognition/csv_to_wav/adv/adv_audio/8k.84-121123-0000.wav" # do you hear
    # path = "/csv_to_wav/adv/8k.84-121123-0012.wav"
    # path = "/data/guofeng/Speech_Recognition/csv_to_wav/adv/adv_audio/sample-000000.wav"
    path = '/data/guofeng/Speech_Recognition/csv_to_wav/adv/8k.84-121123-0000.wav'
    # path = "/data/guofeng/Speech_Recognition/csv_to_wav/adv/adv_audio/adversarial_yuxuangf.wav"
    # path = "/data/guofeng/Speech_Recognition/csv_to_wav/adv/adv_audio/adv.wav"
    # path = "/data/guofeng/audio_adversaril/audio_adversarial_examples/adv.wav"
    # path = '/fgm_data/aliyunAudio.wav'
    str, name = baidu_recog(path)
    # fg = open('test.txt', 'w')
    # fg.write('%s\t%d\tresult:\t%s\n' % (name, 1, str))
    print("result:", str)
    # print("filename:", name)
    # file_path = "/data/guofeng/Speech_Recognition/csv_to_wav/adv/adv_audio"
    # file_path = "/data/guofeng/Speech_Recognition/csv_to_wav/yuxuan24"
    # for path in getfile_path(file_path):            # batch processing
    #     re, na = baidu_recog(path)
    #     print("%s:\t%s" % (na, re))
