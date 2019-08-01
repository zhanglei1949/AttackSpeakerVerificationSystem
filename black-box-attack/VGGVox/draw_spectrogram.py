import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import fbank
def spec(filename, source_dir):
    """
    splited = filename.split('/')
    ori_filename = splited[0] + '/ori/' + splited[2].split('_')[0] + '.wav'
    y1, sr = librosa.load(filename, 16000)
    y2, sr = librosa.load(ori_filename, 16000)
    print(np.max(y1), np.min(y1))
    fig = plt.figure(figsize=(8,3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y1, n_fft = 512, hop_length = 160, win_length = 400)), ref=1)
    #plt.subplot(2,1,1)
    
    #(1025, 162)
    D = D[:256, :160]
    print(D.shape)
    librosa.display.specshow(D, y_axis='linear', sr = 16000, fmax = 8000, hop_length=160, vmax = 30, vmin=-40, cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    #plt.clim(-1,1)
    plt.tight_layout()
    fig.savefig(splited[2].split('.')[0] + '_adv_spec.pdf')
"""
    splited = filename.split('/')
    ori_filename = source_dir + '-'.join(splited[2].split('-')[0:4]) + '.wav'
    print(ori_filename)
    y1, sr = librosa.load(filename, 16000)
    y2, sr = librosa.load(ori_filename, 16000)
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1, n_fft = 512, hop_length = 160, win_length = 400)))
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2, n_fft = 512, hop_length = 160, win_length = 400)))
    plt.subplot(2,1,1)
    librosa.display.specshow(D1, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(2,1,2)
    librosa.display.specshow(D2, y_axis='linear')
    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=160)
    #S, a = fbank(signal = y, nfilt = 128, samplerate=16000,winlen=0.025, winstep=0.01)
    #print(S.shape)
    #figsize=(10,4)
    #librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr = 16000, hop_length = 160, y_axis='mel', fmax=8000, x_axis='time')
    #librosa.display.specshow(D, y_axis='linear', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
#    path_to_save = splited[2].split('_')[0] + '_spec.pdf'
#    fig.savefig(path_to_save)
def waveplot(filename, source_dir):
    splited = filename.split('/')
    ori_filename = source_dir + '-'.join(splited[2].split('-')[0:4]) + '.wav'
    print(ori_filename)
    plt.subplot(2,1,1)
    y1, sr = librosa.load(filename, 16000)
    y2, sr = librosa.load(ori_filename, 16000)
    librosa.display.waveplot(y=y1-y2, sr=sr)
#    plt.colorbar(format='%+2.0f dB')
    plt.subplot(2,1,2)
    librosa.display.waveplot(y=y2, sr=sr)
#    plt.colorbar(format='%+2.0f dB')

    #D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    #plt.subplot(2,1,1)
    plt.tight_layout()
    plt.show()
    #fig.savefig(splited[2].split('.')[0] + '_ori_wav.pdf')

def draw_threshold(filename, npyname = './hearing_threshold_dB.npy'):
    threshold = np.load(npyname)
    splited = filename.split('/')
    ori_filename = splited[0] + '/ori/' + splited[2].split('_')[0] + '.wav'
    y1, sr = librosa.load(filename, 16000)
    y2, sr = librosa.load(ori_filename, 16000)
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1, n_fft = 512, hop_length = 160, win_length = 400)), ref=1)
    D1 = D1[:256, :160]
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2, n_fft = 512, hop_length = 160, win_length = 400)), ref=1)
    D2 = D2[:256, :160]
    threshold = threshold.T
    threshold_spec = 20* np.log10( np.abs(threshold) / (np.max(D2)))
    D_spec = 20* np.log10( np.abs(D2 - D1) / (np.max(D2)))
    dif = D_spec - threshold_spec
    print(dif.shape)
    print(np.mean(dif))
    print(np.count_nonzero(dif > 0))
    fig = plt.figure(figsize=(8,3))
    librosa.display.specshow(threshold_spec, y_axis='linear', sr = 16000, fmax = 8000, hop_length=160, vmax = 30, vmin=-40, cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    #plt.clim(-1,1)
    plt.tight_layout()
    fig.savefig(splited[2].split('.')[0] + '_thr_spec.pdf')

#spec('ori_att_audios_0_8/ori/174-168635-0022-clip.wav', '174_ori.pdf')
#spec('ori_att_audios_0_8/adv/174-168635-0022-clip_1272.wav', '174-1272_adv.pdf')
#wavplot('ori_att_audios_0_8/ori/1272-135031-0023-clip.wav', '1272_wav_ori.pdf')
#wavplot('ori_att_audios_0_8/adv/1272-135031-0023-clip_5338.wav', '1272-5338_adv.pdf')
#wavplot('ori_att_audios_0_8/ori/1272-135031-0023-clip.wav', 'ori_att_audios_0_8/adv/1272-135031-0023-clip_5338.wav', '1272-5338_adv.pdf')
#spec('ori_att_audios_0_8/adv/1673-143396-0008-clip_1919.wav')
#draw_threshold('ori_att_audios_0_8/adv/1673-143396-0008-clip_1919.wav')
spec('./adv/174-50561-0006-clip-251-log.wav', '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/')
#waveplot('./adv/174-50561-0006-clip-251-log.wav', '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/')
#waveplot('./adv/422-122949-0022-clip-251-log.wav', '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/')

