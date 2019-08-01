import glob

from random import sample
ids = ['84', '174', '251', '422', '652']
#files = glob.glob('./adv/')
ori_dir = '/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/'
output = './threshold_trials.txt'
f = open(output, 'w')
for i in ids:
    f1 = glob.glob('./adv/' + i + '*')[0]
    fori = glob.glob(ori_dir + i + '*.wav')
    fothers = set(glob.glob(ori_dir + '*.wav')) - set(fori)

    #10 from same speaker
    same_speakers = sample(fori, 10)
    for j in same_speakers:
        f.write(f1 + ' ' + j + ' ' + str(1) + '\n')
    diff_speakers = sample(fothers, 10)
    for j in diff_speakers:
        f.write(f1 + ' ' + j + ' ' + str(0) + '\n')
    #10 from different speaker
    
f.close()
