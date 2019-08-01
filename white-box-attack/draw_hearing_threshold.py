import matlab
import matlab.engine
import numpy as np
engine = matlab.engine.start_matlab()

res_thr, res_thr_db = engine.test_threshold('/home/lei/2019/AttackSpeakerVerification/white-box-attack/audio2text/ori_att_audios_0_8/ori/1673-143396-0008-clip.wav', nargout=2)
res = np.asarray(res_thr_db)
print(res.shape)
print(np.max(res), np.min(res))
res = res[:160,:]
np.save('hearing_threshold_dB.npy', res)
