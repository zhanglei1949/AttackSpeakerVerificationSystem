import librosa
import numpy as np
import sys
from glob import glob
import soundfile as sf
sys.path.append('../../speakerVerificationSystem/')
import silence_detector
import constants as c
def VAD(audio):
    chunk_size = int(16000*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=16000, target_dir='./'):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = VAD(audio.flatten())
    #start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    #start_frame = int(start_sec * sample_rate)
    #end_frame = int(end_sec * sample_rate)

    #if len(audio) < (end_frame - start_frame):
    #    au = [0] * (end_frame - start_frame)
    #    for i in range(len(audio)):
    #        au[i] = audio[i]
    #    audio = np.array(au)
    # clip
    #audio = audio[:22580]
    audio_length = 25840
    num_splits = int(len(audio) / audio_length) + 1
    for i in range(num_splits):
        path_to_save = target_dir+ filename.split('/')[-1][:-4] + '_' + str(i) + '.wav'
        saved = audio[i*audio_length : (i+1)*audio_length]
        print(len(saved))
        if (len(saved) < audio_length):
            saved = np.concatenate((saved, np.zeros((audio_length - len(saved)))))
        sf.write(path_to_save, saved, 16000, 'PCM_16')
def vad_audio(filename, sample_rate = 16000, target_dir = './'):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = VAD(audio.flatten())

    sf.write(target_dir + filename.split('/')[-1][:-4]  + '_vad.wav', audio, 16000, 'PCM_16')

def stick_together(source_dir, target_dir, person , phrase, num_phrase):
    for i in range(1, num_phrase+1):
        subs = glob(source_dir  +'suibin_' +  phrase + '_' + str(i) + '_*.wav')
        subs.sort()
        print(subs)
        a = []
        for f in subs:
            audio, fs = librosa.load(f, 16000)
            a.append(audio)
        res = np.asarray(a)
        res = np.reshape(res, (res.shape[0] * res.shape[1], 1))
        print(res.shape)
        sf.write(target_dir + person + '_' + phrase + '_' + str(i) + '.wav', res, 16000, 'PCM_16') 
person='Suibin'
source_dir = '../../data/'+ person + '/Gmail/'
target_dir = '../../data/' + person + '/clip/'
files = glob(source_dir + '*ph1*.wav')
#for file in files:
    #audio = read_audio(file, 16000, target_dir)
    #vad_audio(file, 16000, target_dir)
    #read_audio(file, 16000, target_dir)
    #sf.write(target_dir + file.split('/')[-1], audio, 16000, 'PCM_16')

stick_together(target_dir, target_dir, person, 'ph1', 5)
