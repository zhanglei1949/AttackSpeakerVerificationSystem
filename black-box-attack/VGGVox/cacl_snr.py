import glob as glob
import librosa
import numpy as np
import os 
adv_dir = './adv/'
ori_dir = '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/'
wavs = glob.glob(adv_dir + '*.wav')
def cal_snr(fa, fb):
    adv,_ = librosa.load(fa, 16000)
    ori,_ = librosa.load(fb, 16000)
    noise = adv - ori
    p1 = np.power(noise, 2)
    p2 = np.power(ori, 2)
    a1 = np.sqrt(np.mean(p1))
    a2 = np.sqrt(np.mean(p2))
    return 20*np.log10(a2/a1)
    
for audio in wavs:
    splited = audio.split('/')[-1].split('-')
    ori = ori_dir + '-'.join(splited[:4]) + '.wav'
    #print(audio, ori)
    print(cal_snr(audio, ori))

