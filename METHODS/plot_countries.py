import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import sys
import seaborn as sns
import os

rok = '2018'
model_option = "K"
path = 'C:/Informatyka/RES/dataset/'
file="RES_DATASET_" + rok + ".csv"

pathfile = os.path.join(path, file)
dane = pd.read_csv(pathfile)
todraw = dane.iloc[:len(dane)-2,:]

if model_option == "K":
    todraw = todraw.rename(columns = {'k1_num': r'$C_1$', 'k2_num': r'$C_2$', 'k3_num': r'$C_3$', 'k4_num': r'$C_4$'})
else:
    todraw = todraw.rename(columns = {'k1.1': r'$C_{1.1}$', 
                                  'k1.2': r'$C_{1.2}$', 
                                  'k1.3': r'$C_{1.3}$', 
                                  'k1.4': r'$C_{1.4}$',
                                  'k1.5': r'$C_{1.5}$',
                                  'k2.1': r'$C_{2.1}$',
                                  'k2.2': r'$C_{2.2}$',
                                  'k2.3': r'$C_{2.3}$',
                                  'k2.4': r'$C_{2.4}$',
                                  'k3.1': r'$C_{3.1}$',
                                  'k3.2': r'$C_{3.2}$',
                                  'k3.3': r'$C_{3.3}$',
                                  'k4.1': r'$C_{4.1}$',
                                  'k4.2': r'$C_{4.2}$',
                                  'k4.3': r'$C_{4.3}$',
                                  })

lista_ind = []
for i in range(1, 31):
    lista_ind.append(r'$A_{' + str(i) + '}$')
todraw = todraw.set_index('ID')
todraw = todraw / todraw.sum(axis = 0)
sns.set()
ax = todraw.plot(kind='bar', stacked=True)
plt.setp(ax.patches, linewidth=0)
plt.xlabel('Countries')
plt.ylabel('Normalized values of criteria')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('dataset_' + model_option + '_normalized_' + rok + '.pdf')
plt.show()