# Calling hearing threshold calculation via matlab api
import matlab
import matlab.engine
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from python_speech_features import fbank
if __name__ == '__main__':
    audioname = '../../data/adversarial/422-122949-0003-clip-1993.wav'
    #audio, fs = librosa.load(audioname, 16000)
    #spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=160, win_length = 400))
    #librosa.display.specshow(librosa.amplitude_to_db(spec,
    #    ref=np.max),
    #    y_axis='log', x_axis='time')
    #plt.title('Power spectrogram')
    #plt.colorbar(format='%+2.0f dB')
    #plt.tight_layout()
    #plt.show()

    #engine = matlab.engine.start_matlab()
    #audioname = '422-122949-0003-clip-1993.wav'
    #audioname = '00001.wav'
    #threshold= engine.test_threshold(audioname)
    #threshold = np.asarray(threshold)
    #print(threshold.shape)
    
    #python speech feature
    audio, fs = librosa.load(audioname, 16000, mono = True) 
    features, _ = fbank(audio, 16000, nfilt = 64, winlen = 0.025)
    print(features.shape)
