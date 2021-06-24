import numpy as np
import itertools as it
import copy
import pandas as pd

def load_Input(path):
    df = pd.read_csv(path, header=None)
    profcost = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    a,b = df.shape
    tab = df.to_numpy()
    profcost = profcost.to_numpy()
    C = []
    for i in range(a):
        Ci = []
        for j in range(b):
            Ci.append(tab[i,j])
        C.append(Ci)
    return C, profcost


def load_Alternatives(path):
    df = pd.read_csv(path, header=None)
    a,b = df.shape
    tab = df.to_numpy()
    C = []
    for i in range(a):
        Ci = []
        for j in range(b):
            Ci.append(tab[i,j])
        C.append(Ci)
    return C
    

def preference(MEJ):
    SJ = np.sum(MEJ, axis=1)
    SJ_copy = copy.copy(SJ)
    k = len(np.unique(SJ))
    P = np.zeros(len(SJ))
    delta2 = float(0)
    for i in range(k):
        index = np.where(SJ_copy == min(SJ_copy))
        P[index] = float(delta2)
        delta2 += 1 / (k - 1)
        SJ_copy[index] = max(SJ_copy) + 1
    return SJ, P


def tfn(x, a, m, b):
    if x < a or x > b:
        return 0
    elif a <= x < m:
        return (x-a) / (m-a)
    elif m < x <= b:
        return (b-x) / (b-m)
    elif x == m:
        return 1


def alternative(C, x, ind):
    if ind == 0:
        return tfn(x, C[ind], C[ind], C[ind + 1])
    elif ind == len(C) - 1:
        return tfn(x, C[ind - 1], C[ind], C[ind])
    else:
        return tfn(x, C[ind - 1], C[ind], C[ind + 1])

def getMEJ(CO, profcost):
    M = len(CO)
    N = len(CO[0])
    preMEJ = np.zeros(M)
    for i in range(len(CO)):
        suma = 0
        for j in range(len(CO[i])):
            if profcost[j] == 1: #profit
                suma += CO[i][j]
            elif profcost[j] == 0: #cost
                suma -= CO[i][j]
        preMEJ[i] = suma
    MEJ = np.zeros((M,M))
    for i in range(len(preMEJ)):
        for j in range(len(preMEJ)):
            if preMEJ[i] == preMEJ[j]:
                MEJ[i,j] = 0.5
            elif preMEJ[i] > preMEJ[j]:
                MEJ[i,j] = 1
    return MEJ


def main():
    nameInput = "Input2018P.csv"
    nameAlt = "ALT2018P.csv"
    C, profcost = load_Input(nameInput)
    Alty = load_Alternatives(nameAlt)
    
    CO = list(it.product(*C))
    
    MEJ = getMEJ(CO, profcost)

    SJ, P = preference(MEJ)
    
    bazaregul = {str(i): [] for i in range(len(CO[0]) + 2)}
    
    for i in range(len(P)):
        print('IF ' + str(CO[i]) + ' THEN ' + str(P[i]))
        for j in range(len(CO[i])):
            print('CO[i][j]:', CO[i][j])
            bazaregul[str(j)].append(CO[i][j])
        bazaregul[str(j + 1)].append(SJ[i])
        bazaregul[str(j + 2)].append(P[i])

    print('BAZA REGUL')
    bazaRegul = pd.DataFrame(bazaregul)
    print(bazaRegul)
    
    pi_list = []
    for i in range(len(Alty)):
        Alt = Alty[i]
        print('Alternatives: \n' + str(Alt))
        W = []
        Index = []
        Score = 0
        for i in range(len(P)):
            for j in range(len(CO[i])):
                W.append(alternative(C[j], Alt[j], C[j].index(CO[i][j])))
            if np.product(W) * P[i] > 0:
                Index.append(i)
            Score += np.product(W) * P[i]
            W = []
        pi_list.append(Score)
        print('\nActivated rules:')
        for i in Index:
            print('IF ' + str(CO[i]) + ' THEN ' + str(P[i]))
        print('\nScore: \n' + str(Score))
   
    pi_list = np.array(pi_list)
    ind_rank = np.argsort(-pi_list)
    ranking = np.argsort(ind_rank) + 1
    print(ranking)

    scores = {'Pi': pi_list, 'Rank': ranking}
    df_comet = pd.DataFrame(scores)
    df_comet.to_csv('comet_results.csv')
    
    dfMEJ = pd.DataFrame(MEJ)
    dfSJ = pd.DataFrame(SJ)
    dfP = pd.DataFrame(P)
    dfMEJSJP = pd.concat([dfMEJ, dfSJ, dfP], axis=1)
    dfMEJSJP.to_csv('results.csv')
    
main()

