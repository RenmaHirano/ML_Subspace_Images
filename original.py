import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

def GetTrainAve(m :int):
    xtr_ave = np.empty(np.array(X_train[m][0]).size)
    for i in range(np.array(X_train[m][0]).size):
        xtr_ave[i] = np.sum(np.array(X_train[m][:,i]))/400
        # print(np.sum(np.array(X_train[m][:,i]))/400)

    return np.sum(xtr_ave)/np.array(X_train[m][0]).size

def GetTestAve(m :int):
    xtes_ave = np.empty(np.array(X_test[m][0]).size)
    for i in range(np.array(X_test[m][0]).size):
        xtes_ave[i] = np.sum(np.array(X_test[m][:,i]))/400
        # print(np.sum(np.array(X_test[m][:,i]))/400)

    return np.sum(xtes_ave)/np.array(X_test[m][0]).size

def main():
    ave_test_matrix = np.empty((282))
    ave_train_matrix = np.empty((141))
    for s in range(141):
        ave_train_matrix[s] = GetTrainAve(s)
        print(ave_train_matrix[s])
    for t in range(282):
        ave_test_matrix[t] = GetTestAve(t)
    
    ans_vector = np.empty(282)
    check_vector = np.empty(282)
    for j in range(282):
        idx = np.argmin(np.abs(ave_train_matrix - ave_test_matrix[j]))
        ans_vector[j] = idx
        print(j // 6)
        if (j // 6 == ans_vector[j] // 3):
            check_vector[j] = 1
        else:
            check_vector[j] = 0
    print(np.sum(check_vector)/282)
        
    pass

if __name__ == "__main__":
    main()
