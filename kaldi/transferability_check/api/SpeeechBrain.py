import os
os.environ['HUGGINGFACE_HUB_CACHE'] = './pretrained_models/speechbrain'
# import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
# from speechbrain.pretrained import EncoderDecoderASR

def speechbarin_rcog(path):
    filename = path.split('/')[-1]
    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
    #                                            savedir="spretrained_models")

    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech",
                                               savedir="./pretrained_models/asr-crdnn-transformerlm-librispeech")

    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
    #                                            savedir="/home/guofeng/spretrained_models/asr-transformer-transformerlm-librispeech")
    # print(asr_model.modules)
    # print(asr_model.hparams)
    prediction = asr_model.transcribe_file(path)
    result = prediction.lower()
    return result, filename


if __name__ == '__main__':
    path = '/data/guofeng/Speech_Recognition/ori_audio/sample-000000.wav'
    pre,na = speechbarin_rcog(path)
    print(pre)
