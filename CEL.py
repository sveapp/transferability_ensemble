import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import argparse
import math


def getCommandDict():
    label = r'/ ETAONIHSRDLUMWCFGYPBVKXJQZ'
    DICT = {}
    test5 = np.zeros((1, 28))
    count = 5
    for i in range(int(test5.shape[1])):
        test5[0][i] = count
        count += 1
    num = 0
    order = np.array([[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]])
    for j in range(order.shape[1]):
        DICT[label[num]] = order[0][j]
        num += 1
    return DICT


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


# Get the output logits of the input example from the model
def getLogits(x):
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


# Get the subsequence that is more similar to the target command from the output logits
def getTheBestTargetIndexSet(logits_softmax, command, dict, logits) :
    CurrentLogitsIndex = 0
    Threshold = 0.2
    flag = True
    flag2 = False
    maxSumSfm = 0
    maxIndexSet = []
    diterNum = 0
    while flag and (logits_softmax.size(0) - CurrentLogitsIndex) > len(command) :
        CurrentCommandIndex = 0
        SumSfm = 0
        indexSet = []
        for i in range(CurrentLogitsIndex + 1, logits_softmax.size(0)) :
            if logits_softmax[i][dict[command[CurrentCommandIndex]]] > Threshold :
                SumSfm += float(logits_softmax[i][dict[command[CurrentCommandIndex]]])
                indexSet.append(i)
                CurrentCommandIndex += 1
            else:
                if float(torch.argmax(logits[i], dim=-1)) != 0 :
                    SumSfm += float(logits_softmax[i][0])
            if len(indexSet) >= len(command) :
                break
        if len(indexSet) < len(command) :
            Threshold /= (1.1 + math.exp(-diterNum))
            diterNum += 1
            if flag2 :
                flag = False
        else:
            CurrentLogitsIndex = indexSet[0]
            flag2 = True
            if SumSfm > maxSumSfm :
                maxSumSfm = SumSfm
                maxIndexSet = indexSet

    return maxIndexSet


# Obtain the modified cross-entropy loss function
def getf2loss(logit_softmax, command_replaced, dict, logits, maxIndexSet) :
    loss_cel = nn.CrossEntropyLoss()
    loss2 = torch.zeros_like(loss_cel(logits[0], torch.tensor(0, dtype = torch.long, device='cuda:0')))
    commandindex = 0
    for i in range(logit_softmax.size(0)) :
        if commandindex < len(maxIndexSet) and i == maxIndexSet[commandindex] :
            if float(torch.argmax(logits[i], dim=-1)) != dict[command_replaced[commandindex]]:
                loss2 += loss_cel(logits[i], torch.tensor(dict[command_replaced[commandindex]], dtype = torch.long, device='cuda:0')) * (logit_softmax[i][torch.argmax(logits[i], dim=-1)] + (1 - logit_softmax[i][dict[command_replaced[commandindex]]]))
            commandindex += 1
        else :
            if float(torch.argmax(logits[i], dim=-1)) != 0 :
                loss2 += loss_cel(logits[i], torch.tensor(0, dtype=torch.long, device='cuda:0')) * (logit_softmax[i][torch.argmax(logits[i], dim=-1)] + (1 - logit_softmax[i][0]))

    return loss2


def attack(source, command, save, command_out):

    input = source
    w = input
    w.requires_grad = True
    optimizer = torch.optim.Adam([w], lr = 0.0005)

    logits = getLogits(w)
    logits = logits.reshape(-1, 32)
    logits_softmax = torch.softmax(logits, dim=-1)

    command_replaced = command.upper()
    dict = getCommandDict()
    maxIndexSet = getTheBestTargetIndexSet(logits_softmax, command_replaced, dict, logits)

    mse = nn.MSELoss()
    count = 0
    steps = 3000
    for step in range(steps):

        AE_Input = w
        # print('source:', adv_images, input)
        cur_f1loss = mse(AE_Input, input)
        f1loss = cur_f1loss.sum()

        logits = getLogits(AE_Input)
        logits2 = logits
        logits = logits.reshape(-1, 32)
        logits_softmax = torch.softmax(logits, dim=-1)

        f_loss = getf2loss(logits_softmax, command_replaced, dict, logits, maxIndexSet)

        AE_Command = processor.batch_decode(torch.argmax(logits2, dim=-1))
        print('AE_Command:', AE_Command[0])

        if AE_Command[0] == command_out.upper():
            print('succeed')
            torchaudio.save(save, AE_Input.detach().cpu().reshape(1, -1), 16000)
            break

        cost =  f1loss +  f_loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        count += 1

        # print("CostItem: ", cost.item(), "L2_loss: ", L2_loss, "step: ", step)

if __name__ == '__main__':
    # Consecutive letters in the original output command need to be added with '/'
    text = ['Text my client hel/lo', 'Show me the payment code', 'Your lock is broken',
            'Remind me to cal/l Bil/l', 'Turn of/f al/l alarms', 'Turn on automatic renewal',
            'Clear the voice mail',  'Scre/en capture', 'Your flight has be/en canceled',
            'Navigate to Golden Gate Bridge']
    text_out = ['Text my client hello', 'Show me the payment code', 'Your lock is broken',
            'Remind me to call Bill', 'Turn off all alarms', 'Turn on automatic renewal',
            'Clear the voice mail',  'Screen capture', 'Your flight has been canceled',
            'Navigate to Golden Gate Bridge']
    processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')
    model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a')

    parser = argparse.ArgumentParser(description='Adversarial Examples')
    parser.add_argument('--originalFile', dest='originalFile', type=str, help='Addreess of the sourceOriginal file', default=r'D:\ZWJ\TrustWorthy\data\music\01.wav')
    parser.add_argument('--command', dest='command', type=str, help='Attack target command', default=text[0])
    parser.add_argument('--save', dest='save', type=str, help='Addreess of the saved file', default='./AES/carrier01_c01.wav')
    parser.add_argument('--command_out', dest='command_out', type=str, help='Compare target command', default=text_out[0])
    args = parser.parse_args()

    y_torch, sr_tar = torchaudio.load(args.originalFile, channels_first=False)
    y_torch = y_torch.squeeze(dim=1)
    y_torch.requires_grad = True

    command = text[int(args.command)]
    command_out = text_out[int(args.command)]
    print(int(args.command))
    attack(source=y_torch, command=command, save=args.save, command_out=command_out)
