import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

def GetTrainSubspace(m :int):
    pca_1 = PCA()
    first_x_train = np.array(X_train[m][:,0])
    xtr = np.array(first_x_train)
    for i in range(np.array(X_train[m][0]).size):
        if i != 0:
            xtr = np.vstack([xtr, np.array(X_train[m][:,i])])
    pca_1.fit(xtr)
    subspace_1 = np.dot(pca_1.components_.T, pca_1.components_)
    
    return subspace_1

def GetTestSubspace(m :int):
    pca_2 = PCA()
    first_x_test = np.array(X_test[m][:,0])
    xtes = np.array(first_x_test)
    for i in range(np.array(X_test[m][0]).size):
        if i != 0:
            xtes = np.vstack([xtes, np.array(X_test[m][:,i])])
    pca_2.fit(xtes)
    subspace_2 = np.dot(pca_2.components_.T, pca_2.components_)
    
    return subspace_2

def main():
    last_eig_matrix = np.empty((282, 141))
    for i in range(282):
        temp_matrix = GetTestSubspace(i)
        print("now processing " + str(i))
        for j in range(141):
            for_eig = np.dot(temp_matrix, GetTrainSubspace(j))
            values = np.linalg.eigvals(for_eig)
            values = values.real
            last_eig_matrix[i,j] = abs(values).max()
    
    np.savetxt('np_save.csv', last_eig_matrix, fmt='%.5e', delimiter=",")
    
    pass

if __name__ == "__main__":
    main()
