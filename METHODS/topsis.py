#The TOPSIS Method
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import sys
import seaborn as sns
import os


def topsis(dane, weight_type):
    macierz = dane.iloc[:len(dane)-2,1:]
    X = macierz.to_numpy()
    costs = dane.iloc[len(dane)-1,1:].to_numpy()
    
    if weight_type == "TOPSIS equal":
        wagi = dane.iloc[len(dane)-2,1:].to_numpy()
    elif weight_type == "TOPSIS entropy":
        wagi = entropy(X)
    
    X = normalization_mini_max(X, costs)
    V = X * wagi

    PIS = np.amax(V, axis=0)
    NIS = np.amin(V, axis=0)

    DP = (V-PIS)**2
    DP = np.sum(DP, axis=1)
    DP = DP**0.5

    DM = (V-NIS)**2
    DM = np.sum(DM, axis=1)
    DM = DM**0.5

    C = DM/(DM+DP)
    
    return C

def normalization_max(X, criteria_type):
    maximes = np.amax(X, axis=0)
    minimums = np.amin(X, axis=0)
    ind = np.where(criteria_type == 0)
    X = X/maximes
    X[:,ind] = 1-X[:,ind]
    return X


def normalization_mini_max(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == 0)

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(X[:, ind_profit], axis = 0)
                             ) / (np.amax(X[:, ind_profit], axis = 0) - np.amin(X[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(X[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(X[:, ind_cost], axis = 0) - np.amin(X[:, ind_cost], axis = 0))

    return x_norm



def entropy(X):
    pij = X / np.sum(X, axis = 0)
    Ej = - np.sum((pij * np.log10(pij + sys.float_info.epsilon)), axis = 0) / (np.log10(X.shape[0]))
    wagi = (1 - Ej) / (np.sum(1 - Ej))
    return wagi



rok = '2018'
model_option = "K"
path = 'C:/Informatyka/magisterkaRES/dataset/'
file="RE_DATASET_" + rok + "_datasetTW" + model_option + ".csv"

pathfile = os.path.join(path, file)
dane = pd.read_csv(pathfile)

df_writer = pd.DataFrame()
weight_types = ["TOPSIS entropy"]
for weight_type in weight_types:
    C = topsis(dane, weight_type)
    rankingPrep = np.argsort(-C)
    ranking = np.argsort(rankingPrep) + 1

    df_writer[weight_type] = C
    df_writer[weight_type + ' rank'] = ranking

df_writer.to_csv('nowewyniki' + rok + '_' + model_option + '.csv')