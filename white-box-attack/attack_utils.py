import os
import subprocess
import glob
import librosa
import math
import tensorflow as tf
import numpy as np
import soundfile
# Assume the audio is filtered, and fixed length.

from pre_process import read_audio,extract_features
from constants import SAMPLE_RATE
from keras.engine.topology import Layer
import keras.backend as K
#from select_batch import clipped_audio
import constants as c

import keras
from keras import optimizers 
#from models import my_convolutional_model 
import numpy as np
from keras.models import Model
#import tensorflow as tf
from keras import backend as K
from keras.layers import Input
#self defined noise layer
#from attack_utils import add_noise, fbank_layer

K.set_learning_phase(1)


from python_speech_features import get_filterbanks
#def extract_fbank():
# return 1
# define a new keras layer, 

# define a custimized cosine distance loss function
def load_wavs(self, dataset_dir):
    vox = pd.DataFrame()
    vox['wav'] = glob.glob(dataset_dir + '**/*.wav')
    vox['speaker_id'] = vox['wav'].apply(lambda x : x.split('/')[-1].split('-')[0])
    num_speakers = len(vox['speaker_id'].unique())
    print("Load {} wavs from {} speakers".format(str(len(vox)), str(num_speakers)))
    return vox

def cosineDistance(y_true, y_pred):
    y_true_norm = K.l2_normalize(y_true, axis = -1)
    y_pred_norm = K.l2_normalize(y_pred, axis = -1)
    #return K.dot(y_true_norm, y_pred_norm)
    return 1- K.mean(K.dot(y_true_norm, K.transpose(y_pred_norm)))
def cosineDistanceLoss():
    def cosine_distance_cal(y_true, y_pred):
        return cosineDistance(y_true, y_pred)
    return cosine_distance_cal
class add_noise(Layer):
    def __init__(self, **kwargs):
        super(add_noise, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='noise',
                                    shape=(input_shape[1],),
                                 initializer='zeros',  # TODO: Choose your initializer
                                 trainable=True)
        super(add_noise, self).build(input_shape)

    def call(self, x, **kwargs):
        return x + self.kernel
    def compute_output_shape(self, input_shape):
        # output shape is (?, 25840)
        return input_shape
class fbank_layer(Layer):
    def __init__(self, **kwargs):
        super(fbank_layer, self).__init__(**kwargs)
    def build(self, input_shape):
        
        self.winlen = 0.025
        self.hop_length = 0.01
        self.nfft = 512
        self.nfilt = 64
        # Calculate input shape
        #self.output_shape=(160,64)
        super(fbank_layer, self).build(input_shape)

    def call(self, x, **kwargs):
        #1. preemphasis
        x = K.concatenate( [x[:,:1], x[:,1:] - 0.97*x[:,:-1]], axis = 1)
        #2. padding
        audio_len = int(x.shape[1])
        frame_len = int(c.SAMPLE_RATE * self.winlen)
        frame_step = int(c.SAMPLE_RATE * self.hop_length)

        num_frames =  1 + int(math.ceil((1.0 * audio_len - frame_len)/frame_step))
        #print('num_frames', num_frames)
        paddings = K.zeros((1, int((num_frames - 1) * frame_step + frame_len) - audio_len), dtype=tf.float32)
        padded = K.concatenate( [x, paddings] , axis = 1)
        #print('padded', padded.shape)

        #3. windowing tnto frames
        windowed = K.stack([padded[:,i : i + frame_len] for i in range(0, audio_len - frame_len + 1, frame_step)], 1)

        #4. take fft, to frequency space
        fft = tf.spectral.rfft(windowed, [self.nfft])
        fft = 1.0 / self.nfft * tf.square(tf.abs(fft))

        #5. compute the mel features of fft
        energy = tf.reduce_sum(fft, axis=2)+1e-30
        filters = get_filterbanks(self.nfilt, self.nfft, c.SAMPLE_RATE, 0, None).T
        feat = tf.matmul(fft, np.array([filters]*1, dtype = np.float32)) + 1e-30
        #6. Scale
        outlist = []
        for i in range(0, feat.shape[1]):
            outlist.append(tf.reshape(tf.div(
                    tf.subtract(feat[:,i,:], tf.reduce_mean(feat[:,i,:])),
                    K.std(feat[:,i,:])
                ), (64, 1)))
        feat = tf.stack(outlist)
        
        feat = tf.reshape(feat, (1, feat.shape[0], feat.shape[1], feat.shape[2]))
       
        return feat
    def compute_output_shape(self, input_shape):
        #
        #return self.output_shape
        return (input_shape[0], 160, 64, 1)


# need VAD first.
def VAD_and_save(source_path, target_path, target_len = 25840):
    # do vad
    audio = read_audio(source_path) # vad doneinside
    if (len(audio) > target_len):
        a = int( (len(audio) - target_len)/2 )
        audio = audio[a : a + target_len]
    elif (len(audio < target_len)):
        return
    soundfile.write(target_path, audio, c.SAMPLE_RATE, subtype='PCM_16')
    #librosa.output.write_wav(target_path, audio, c.SAMPLE_RATE)
    #print('save '+ target_path)
    #features_1 = extract_features(audio, target_sample_rate = SAMPLE_RATE) 
    #features_1 = clipped_audio(features_1)
    #print('features 1'+ str(features_1.shape))
    #features_2 = my_extract_features(audio, target_sample_rate = SAMPLE_RATE, nfilt = 64, winlen = 0.025)

def get_wavs(dir, pattern):
    res = glob.glob(os.path.join(dir, pattern))
    return res
def do_vad(source_dir, target_dir):
    #(status, ids) = commands.getstatusoutput("ls "+source_dir)
    ids = subprocess.check_output(["ls", source_dir] ,shell = False).decode()
    ids = ids.strip().split("\n")
    assert len(ids) == 40
    for id in ids:
        source_sub_dir = os.path.join(source_dir, id)
        #target_sub_dir = os.path.join(target_dir, id)
        '''
        if not os.path.exists(target_sub_dir):
            os.mkdir(target_sub_dir)
        else:
            print(target_sub_dir+" exists.")
        '''
        wavs = get_wavs(source_sub_dir, "*/*.wav")
        print('found ' + str(len(wavs)) + ' wavs')
        # clipping.
        for i in range(len(wavs)):
            wav = wavs[i]
            target_wav_path = target_dir + id + '_' + wav.split('/')[-2] + '_' + wav.split('/')[-1]
            VAD_and_save(wav, target_wav_path)
        #break
    return 1
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

def cal_snr(sig, noise):
    p1 = np.power(sig, 2)
    p2 = np.power(noise, 2)
    a1 = np.sqrt(np.mean(p1))
    a2 = np.sqrt(np.mean(p2))
    return 20*np.log10(a1/a2)
if __name__ == '__main__':
    source_dir = '/home/lei/dataset/voxceleb2/vox/vox1_test/wav/'
    target_dir = '/home/lei/2019/dataset/vox-test-wav-vad/'
    do_vad(source_dir, target_dir)
