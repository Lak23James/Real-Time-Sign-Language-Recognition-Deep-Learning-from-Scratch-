#importing all the required libraries
import pandas as pd
import numpy as np
#Data pre processing step
#Training Data
data_train = pd.read_csv('sign_mnist_train.csv')
print(data_train.head())
data_train=np.array(data_train)
m, n = data_train.shape
np.random.shuffle(data_train)
data_train=data_train.T
print(data_train[:5])
Y_train=data_train[0]
X_train=data_train[1:n]
X_train=X_train/255
#Testing Data
data_test = pd.read_csv('sign_mnist_test.csv')
print(data_test.head())
data_test=np.array(data_test)
i,j= data_test.shape
np.random.shuffle(data_test)
data_test=data_test.T
print(data_test[:5])
Y_test=data_test[0]
X_test=data_test[1:j]
X_test=X_test/255




