import pickle
from matplotlib import pyplot as plt


data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train.shape)

plt.imshow(X_train[0][:, 10].reshape((20, 20)))
plt.show()
