import matlab
import matlab.engine
fname = 'a.wav'
engine = matlab.engine.start_matlab()

res1 = test_get_input(fname, 
