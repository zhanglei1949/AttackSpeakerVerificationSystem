# calculate the cosine similary between two vectors of shape (1,512)
import numpy as np
import glob
from scipy import spatial
def cal_sim(a, b):
    return 1 - spatial.distance.cosine(a, b)
if __name__ == '__main__':
    dir = '../data/embeddings/'
    files = glob.glob(dir + '**.npy')
    npys = [np.load(f) for f in files]
    metrics = np.zeros((len(npys), len(npys)))
    for i in range(len(npys)):
        for j in range(i, len(npys)):
            metrics[i][j] = cal_sim(npys[i], npys[j])
    for i in range(len(metrics)):
        for j in range(len(metrics[i])):
            print("%.3f "%(metrics[i][j]), end = '')
        print()
