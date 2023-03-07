import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch.nn as nn
import torchaudio
import numpy as np
import torch
import shutil
import argparse

torch.manual_seed(1000)

def getLabelDICT():
    label = r' ETAONIHSRDLUMWCFGYPBVKXJQZ'
    DICT = {}
    test5 = np.zeros((1, 27))
    count = 5
    for i in range(int(test5.shape[1])):
        test5[0][i] = count
        count += 1
    num = 0
    order = np.array([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]])
    for j in range(order.shape[1]):
        DICT[label[num]] = order[0][j]
        num += 1
    return DICT

# the preproduce of the Wav2Vec
def norm_mean(input_value, attention_mask, padding_value=0.0):
    """
    Every array in the list is normalized to have zero mean and unit variance
    """
    if attention_mask is not None:
        normed_input_values = []

        for vector, length in zip(input_value, attention_mask.sum(-1)):
            normed_slice = (vector - vector[:length].mean()) / torch.sqrt(vector[:length].var() + 1e-7)
            if length < normed_slice.shape[0]:
                normed_slice[length:] = padding_value
            normed_input_values.append(normed_slice)
    else:
        normed_input_values = [(x - x.mean()) / torch.sqrt(x.var() + 1e-7) for x in input_value]
    return normed_input_values

# Get the output logits of the input example from Wav2Vec
def output(x):
    model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a')
    processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')
    sample_rate = processor.feature_extractor.sampling_rate
    # with torch.no_grad():

    model_inputs = processor(x, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    AM = model_inputs.attention_mask.to(torch.int32)
    AM_arange = [array for array in AM][0].reshape(1, -1)

    after_norm = norm_mean(input_value=[x], attention_mask=AM_arange)[0].reshape(1, -1)
    logits = model(after_norm, attention_mask=AM_arange).logits.cuda()

    return logits

def attack(x, command, save):
    input = x
    flag = 0

    w = input
    w.requires_grad = True
    best_adv_images = input
    best_L2 = 1e10 * torch.ones((len(input)))
    prev_cost = 1e10

    MSELoss = nn.MSELoss(reduction='none')

    #lr = 0.01
    optimizer = torch.optim.Adam([w])
    loss = nn.CTCLoss()
    dict = getLabelDICT()
    count = 0
    steps = 2000
    tar_CTC = []
    for j in command.upper():
        tar_CTC.append(dict[j])
    tar_CTC = torch.Tensor(tar_CTC).to(torch.int32)
    for step in range(steps):

        # Get adversarial images
        # adv_images = self.tanh_space(w)
        adv_images = w

        # Calculate loss part1
        current_L2 = MSELoss(adv_images, input)
        L2_loss = current_L2.sum()

        # Calculate the softmax of the output
        logits = output(adv_images)
        logits_softmax = torch.softmax(logits, dim=-1)

        # loss part2 CTC
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        input_len = torch.Tensor(np.array([logits_softmax.shape[1]]).astype("int64")).to(torch.int32)
        tar_len = torch.Tensor(np.array([tar_CTC.shape[0]]).astype("int64")).to(torch.int32)
        f_loss = loss(log_probs, tar_CTC, input_len, tar_len)

        res = processor.batch_decode(torch.argmax(logits, dim=-1))
        print('Transcription:', res[0])
        if res[0] == command.upper():
            print('SUCCESSFUL!!!!!')
            flag = 1
            torchaudio.save(save + '.wav', adv_images.detach().cpu().reshape(1, -1), 16000)
            break
        # torchaudio.save('./test_' + save + '.wav', adv_images.detach().cpu().reshape(1, -1), 16000)


        c = 0.5
        cost = (1 - c) * L2_loss + c * f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        count += 1

        if step % (steps // 10) == 0:
            prev_cost = cost.item()
            print(prev_cost)

    if flag == 0:
        torchaudio.save(save + '_failed.wav', adv_images.detach().cpu().reshape(1, -1), 16000)

def attckAndSave(source, command, save):
    print("Hi I'm there!")
    y_torch, sr_tar = torchaudio.load(source, channels_first=False)
    y_torch = y_torch.squeeze(dim=1)
    y_torch.requires_grad = True
    attack(y_torch, command, save)

if __name__ == '__main__':
    
    processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')
    model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a')

    parser = argparse.ArgumentParser(description='OPT Adversarial Examples')
    parser.add_argument('--source', dest='source', type=str, help='Addreess of the source file', default=r'./music/01.wav')
    parser.add_argument('--command', dest='command', type=str, help='Your attack target command', default='DanMerry')
    parser.add_argument('--save', dest='save', type=str, help='Addreess of the saved file', default='./generated/test_adv.wav')
    args = parser.parse_args()
    print(args.source)
    print(args.command)
    print(args.save)
    attckAndSave(source=args.source, command=args.command.replace('_', ' '), save=args.save)
    
    '''
    files = os.listdir(base)
    for file in files:
        File = os.path.join(base, file)
        count = 0
        for comm in text:
            count += 1
            y_torch, sr_tar = torchaudio.load(File, channels_first=False)
            y_torch = y_torch.squeeze(dim=1)
            y_torch.requires_grad = True
            if count != 10:
                SAVE = 'carrier' + file.split('.')[0] + '_C0' + str(count)
            else:
                SAVE = 'carrier' + file.split('.')[0] + '_C10'
            print(SAVE, comm)
            attack(y_torch, comm, SAVE)

        print(File)
        '''

