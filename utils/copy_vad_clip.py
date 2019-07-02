import glob
import numpy as np
import os
import soundfile

import sys
sys.path.append('../speakerVerificationSystem/')
from pre_process import read_audio
import constants as c

def VAD_and_save(source_path, target_path, target_len = 25840):

    # do vad
    audio = read_audio(source_path) # vad doneinside
    if (len(audio) > target_len):
        a = int( (len(audio) - target_len)/2 )
        audio = audio[a : a + target_len]
    elif (len(audio < target_len)):
        return
    soundfile.write(target_path, audio, c.SAMPLE_RATE, subtype='PCM_16')

def get_wavs(dir, pattern):
    res = glob.glob(os.path.join(dir, pattern))
    return res
def do_vad(source_dir, target_dir):

    wavs = get_wavs(source_dir, "*/*/*.wav")
    print("obtain %d wavs files" % len(wavs))
    for i in range(len(wavs)):
        wav = wavs[i]
        basename = wav.split('/')[-1]
        new_basename = basename[:-4] + '-clip.wav'
        target_wav_path = target_dir + new_basename
        VAD_and_save(wav, target_wav_path)
        if (i%100 == 0):
            print(wav, target_wav_path)
def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

if __name__ == '__main__':
    source_dir = '/home/lei/d/LibriSpeech/dev-clean/'
    target_dir = '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/'
    do_vad(source_dir, target_dir)
