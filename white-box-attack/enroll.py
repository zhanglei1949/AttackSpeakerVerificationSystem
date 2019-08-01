# For attacking MSV, we need to carry text-dependent enrollment.
# assuming the audio has been preprocess and is 22580 in 16000Hz
import numpy as np
import librosa
import keras
import sys
sys.path.append('../speakerVerificationSystem/')
import tensorflow as tf
import constants as c
import keras.backend as K
import pandas as pd
import soundfile
from glob import glob
from models import convolutional_model
from eval_metrics import evaluate
#from test_model import create_test_data
from python_speech_features import fbank
from pre_process import normalize_frames, read_audio
def enroll(enroll_speaker='xi',embedding_dir='../data/embeddings/',  num_enroll=3, source_dir='../data/', phrase='ph1'):
    checkpoint_path = '../speakerVerificationSystem/checkpoints/model_17200_0.54980.h5'
    wav_dir = source_dir + './' + enroll_speaker + '/wav/'
    wavs = glob(wav_dir + '*' + phrase + '_*.wav')
    assert len(wavs) >= num_enroll
    
    #Load model
    model = convolutional_model()
    model.load_weights(checkpoint_path)
    features_list = []
    for wav in wavs:
        audio = read_audio(wav, 16000)
        audio = audio[:25840]
        features, _ = fbank(audio, 16000, nfilt=64, winlen = 0.025)
        features = normalize_frames(features)
        features_list.append(features)

    #features = np.asarray((features_list))
    inputs = np.asarray(features_list)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2] , 1))
    print(inputs.shape)
    embeddings = model.predict_on_batch(inputs)
    embedding = np.mean(embeddings, axis = 0)
    #np.save(embedding_dir + enroll_speaker + '_' + phrase + '.npy', embedding)
    np.save(embedding_dir + enroll_speaker + '_' + phrase + '.npy', embedding)
#for i in range(1, 11):
#    enroll(phrase='ph' + str(i))

def enroll_digits(enroll_speaker='sun',embedding_dir='../data/embeddings_digits/',  num_enroll=3, source_dir='../data/digits/', phrase='ph1'):
    checkpoint_path = '../speakerVerificationSystem/checkpoints/model_17200_0.54980.h5'
    wav_dir = source_dir + './' + enroll_speaker + '/'
    wavs = glob(wav_dir + '*.wav')
    assert len(wavs) >= num_enroll
    
    #Load model
    model = convolutional_model()
    model.load_weights(checkpoint_path)
    features_list = []
    for wav in wavs:
        #audio = read_audio(wav, 16000)
        audio,fs = librosa.load(wav, 16000)
        #assert len(audio) >= 25840
        print(len(audio))
        if (len(audio) < 25840):
            audio = np.concatenate((audio, np.zeros((25840 - len(audio), ))))
        audio = audio[:25840]
        features, _ = fbank(audio, 16000, nfilt=64, winlen = 0.025)
        features = normalize_frames(features)
        features_list.append(features)

    #features = np.asarray((features_list))
    inputs = np.asarray(features_list)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2] , 1))
    print(inputs.shape)
    embeddings = model.predict_on_batch(inputs)
    embedding = np.mean(embeddings, axis = 0)
    #np.save(embedding_dir + enroll_speaker + '_' + phrase + '.npy', embedding)
    np.save(embedding_dir + enroll_speaker + '_' + phrase + '.npy', embedding)
def enroll_siri(enroll_speaker='yan',embedding_dir='../data/embeddings_siri/',  num_enroll=3, source_dir='../data/siri/', phrase='ph1'):
    checkpoint_path = '../speakerVerificationSystem/checkpoints/model_17200_0.54980.h5'
    wav_dir = source_dir + './' + enroll_speaker + '/'
    wavs = glob(wav_dir + '*.wav')
    assert len(wavs) >= num_enroll
    
    #Load model
    model = convolutional_model()
    model.load_weights(checkpoint_path)
    features_list = []
    for wav in wavs:
        #audio = read_audio(wav, 16000)
        audio,fs = librosa.load(wav, 16000)
        #assert len(audio) >= 25840
        print(len(audio))
        if (len(audio) < 25840):
            audio = np.concatenate((audio, np.zeros((25840 - len(audio), ))))
        audio = audio[:25840]
        features, _ = fbank(audio, 16000, nfilt=64, winlen = 0.025)
        features = normalize_frames(features)
        features_list.append(features)

    #features = np.asarray((features_list))
    inputs = np.asarray(features_list)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2] , 1))
    print(inputs.shape)
    embeddings = model.predict_on_batch(inputs)
    embedding = np.mean(embeddings, axis = 0)
    #np.save(embedding_dir + enroll_speaker + '_' + phrase + '.npy', embedding)
    np.save(embedding_dir + enroll_speaker + '.npy', embedding)
#enroll(phrase='ph')
#enroll_digits(phrase="digits")
enroll_siri()
