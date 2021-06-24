import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product, combinations
import seaborn as sns
from pandas.plotting import scatter_matrix
import os
import matplotlib

rok = '2018'
model_option = 'K'
path = 'C:/Informatyka/RES/dataset'
file = 'RES_RESULTS' + rok + '.csv'
pathfile = os.path.join(path, file)
data = pd.read_csv(pathfile)

weight_types = ["TOPSIS equal", "TOPSIS entropy",
                "COMET"]

list_wt = []

for el in weight_types:
    list_wt.append(el + ' rank')

lista_kombinacji = list(combinations(list_wt, 2))

alternatives_text = []

for i in range(1, 31):
    alternatives_text.append(r'$A_{' + str(i) + '}$')
	
	
plt.style.use('seaborn')
plt.figure(figsize=(16, 5))
list_range = np.arange(0, 35, 5)
for it, el in enumerate(lista_kombinacji):
    
    xx = [min(data[el[0]]), max(data[el[0]])]
    yy = [min(data[el[1]]), max(data[el[1]])]

    ax = plt.subplot(1, 3, it + 1)
    ax.plot(xx, yy, color = 'lightblue', linestyle = '--', zorder=1)

    ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder=2)
    for i, txt in enumerate(alternatives_text):
        ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                     verticalalignment='top', horizontalalignment='left')

    ax.set_xlabel(el[0][:-5], fontsize = 14)
    ax.set_ylabel(el[1][:-5], fontsize = 14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(list_range)
    ax.set_yticks(list_range)

    y_ticks = ax.yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)

    x_ticks = ax.xaxis.get_major_ticks()
    x_ticks[0].label1.set_visible(False)
    ax.set_xlim(-1, 33)
    ax.set_ylim(-1, 31)

    ax.grid(True)
    ax.set_axisbelow(True)
    
plt.tight_layout()
plt.savefig(model_option + rok + '.pdf')
plt.show()