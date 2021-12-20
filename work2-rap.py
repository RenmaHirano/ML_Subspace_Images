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
    
    xtr = np.empty((400, np.array(X_train[m][0]).size))
    Lap_tr = np.empty(400)
    
    for p in range(np.array(X_train[m][0]).size):
        for q in range(400):
            Lap_tr[q] = X_test[m][GetLabelNumForLaplace(q-1),p] + X_test[m][GetLabelNumForLaplace(q-20),p] +X_test[m][GetLabelNumForLaplace(q+1),p] +X_test[m][GetLabelNumForLaplace(q+20),p] - 4 * X_test[m][GetLabelNumForLaplace(q),p]
        xtr[p] = Lap_tr
        
    pca_1.fit(xtr)
    subspace_1 = np.dot(pca_1.components_.T, pca_1.components_)
    
    return subspace_1

def GetTestSubspace(m :int):
    pca_2 = PCA()
    
    xtes = np.empty((400, np.array(X_test[m][0]).size))
    Lap_tes = np.empty(400)
    
    for p in range(np.array(X_test[m][0]).size):
        for q in range(400):
            Lap_tes[q] = X_test[m][GetLabelNumForLaplace(q-1),p] + X_test[m][GetLabelNumForLaplace(q-20),p] +X_test[m][GetLabelNumForLaplace(q+1),p] +X_test[m][GetLabelNumForLaplace(q+20),p] - 4 * X_test[m][GetLabelNumForLaplace(q),p]
        xtes[p] = Lap_tes
        
    pca_2.fit(xtes)
    subspace_2 = np.dot(pca_2.components_.T, pca_2.components_)
    
    return subspace_2
    
def GetLabelNumForLaplace(l :int):
    if (l < 0):
        return 0
    if (l > 400):
        return 400

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
    
    np.savetxt('np_save2.csv', last_eig_matrix, fmt='%.5e', delimiter=",")
    
    pass

if __name__ == "__main__":
    main()
