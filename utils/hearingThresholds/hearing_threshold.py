# Calling hearing threshold calculation via matlab api
import matlab
import matlab.engine
import numpy as np
if __name__ == '__main__':
    engine = matlab.engine.start_matlab()
    audioname = '422-122949-0003-clip-1993.wav'
    #audioname = '00001.wav'
    threshold= engine.test_threshold(audioname)
    threshold = np.asarray(threshold)
    print(threshold.shape)