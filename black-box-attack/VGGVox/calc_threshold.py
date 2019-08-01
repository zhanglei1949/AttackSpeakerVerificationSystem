import matlab
import matlab.engine

engine = matlab.engine.start_matlab()
trial_f = './threshold_trials.txt'
in_ = open(trial_f, 'r')
l = []
output = './verfication_res.txt'
f = open(output, 'w')
for line in in_.readlines():
    filea = line.split(' ')[0]
    fileb = line.split(' ')[1]
    truelabel = line.split(' ')[2]
    res = engine.demo_vggvox_verif_voxceleb2(filea, fileb, nargout=1)
    #l.append(res)
    f.write(truelabel + ' ' + str(res) + '\n')
f.close()
