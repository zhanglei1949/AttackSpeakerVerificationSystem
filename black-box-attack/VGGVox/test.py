import matlab
import matlab.engine
import librosa
import time
import numpy as np
filename = './ori/84-121123-0001-clip.wav'
y,fs = librosa.load(filename, 16000)

engine = matlab.engine.start_matlab()
y2, fs2 = engine.test_getinput(y, nargout=2)
print(np.mean(y-y2))