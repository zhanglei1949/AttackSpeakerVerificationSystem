import numpy as np
f = open('verfication_res.txt', 'r')
lines = f.readlines()
lines = [ line.lstrip().rstrip() for line in lines]

thresholds = np.arange(0, 0.4, 0.001)

best_th = 0
best_err = 10
sum = 0
arr0 = [0] * len(lines)
for i, threshold in enumerate(thresholds):
    for j in range(len(lines)):
        arr0[j] = 
