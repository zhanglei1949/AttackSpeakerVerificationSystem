import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
fig,ax = plt.subplots()
#ori_distance = [
##        [0.83,  0.85,   0.78,   0.80,   1],
 #       [0.86,  0.82,   0.78,   1,      0.80],
#        [0.82,  0.96,   1,      0.78,   0.78],
#        [0.91,  1,      0.95,   0.82,   0.85],
#        [1,     0.91,   0.82,   0.86,   0.83]
#        #1      2       3       4       5
#        ]
adv_distance = [
                #target
        #source
        [0, 0.53, 0.39, 0.48, 0.36],
        [0.25, 0, 0.23, 0.25, 0.20],
        [0.43, 0.51, 0, 0.42, 0.37],
        [0.51, 0.33, 0.44, 0, 0.38],
        [0.40, 0.38, 0.58, 0.34, 0]
]
ori_distance = [
        [0, 0.91, 0.82, 0.86, 0.83],
        [0.91, 0, 0.95, 0.82, 0.85],
        [0.82, 0.95, 0, 0.78, 0.78],
        [0.86, 0.82, 0.78, 0, 0.80],
        [0.83, 0.85, 0.78, 0.80, 0]
]
ori_score = np.ones((5,5)) - ori_distance
#adv_distance = [
#        [0.36,  0.19,   0.37,   0.38,   1],
#        [0.48,  0.25,   0.42,   1,      0.34],
#        [0.39,  0.23,   1,      0.44,      0.58],
#        [0.53,  1,      0.52,   0.33,      0.38],
#        [1,     0.25,   0.43,   0.51,   0.40]
#        ]
adv_score = np.ones((5,5)) - adv_distance
ylabels = ['id-84', 'id-174', 'id-251', 'id-422', 'id-652']
xlabels = ['id-84', 'id-174', 'id-251', 'id-422', 'id-652']
#ylabels = ['id-84', 'id-174', 'id-251', 'id-422', 'id652']
#ylabels = [ 'id-652', 'id-422','id-251', 'id-174', 'id-84']
#cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
#cmpa = sns.cubehelix_palette(8, as_cmap = True)
mask = [
        [0,  0,   0,   0,   1],
        [0,  0,   0,   1,   0],
        [0,  0,   1,   0,   0],
        [0,  1,   0,   0,   0],
        [1,  0,   0,   0,   0]
        ]
#plt.figure(figsize=(10,10))
#sns.heatmap(adv_score, linewidth = 0.15, annot = True, mask = mask, cmap = 'Greys' , vmax=1, vmin=0)
sns.heatmap(adv_score, linewidth = 0.2, annot = True, cmap = 'Greys' , vmax=1, vmin=0, cbar=0, square = 1, annot_kws={'fontsize' : 16})
#ax.set_xticks(np.arange(5))
#ax.set_yticks(np.arange(5))
ax.set_xticklabels(xlabels, fontdict={'fontsize':12})
#ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels, fontdict={'fontsize':12})
#ax.set_yticklabels(ylabels)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', which='both', labelsize=10)
plt.xlabel('Target speaker', fontdict={'fontsize':20}, labelpad=10)
plt.ylabel('Source speaker', fontdict={'fontsize':20}, labelpad=10)
#plt.xticks([])
#plt.yticks([])
#plt.gca().invert_yaxis()
fig.savefig('adv_score_grey_1.pdf', bbox_inches='tight')
#print((np.sum(adv_score) - 5)/20)
#print((np.sum(ori_score) - 5)/20)
