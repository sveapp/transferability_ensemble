import os
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/guofeng/Speech_Recognition/transferability_check/api/pretrained_models/speechbrain'
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                           savedir="/data/guofeng/Speech_Recognition/transferability_check/api/pretrained_models/asr-crdnn-rnnlm-librispeech")

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech",
                                           savedir="/data/guofeng/Speech_Recognition/transferability_check/api/pretrained_models/asr-crdnn-transformerlm-librispeech")
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                           savedir="/data/guofeng/Speech_Recognition/transferability_check/api/pretrained_models/asr-transformer-transformerlm-librispeech")

pred = asr_model.transcribe_file('/data/guofeng/Speech_Recognition/ori_audio/sample-000000.wav')

print(pred.lower())
