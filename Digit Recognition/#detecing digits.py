#detecing digits from mnist data set using neural network from scratch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'train.csv')
mnistData = pd.read_csv(file_path)
#print(mnistData.head())

mnistD = np.array(mnistData)
m, n = mnistD.shape

dtest = mnistD[0:2000].T #now the top row is the labels, and each column is one image
Y_test = dtest[0]
X_test = dtest[1:n]
X_test = X_test / 255.

dtrain = mnistD[2000:m].T
Y_train = dtrain[0]
X_train = dtrain[1:n]
X_train = X_train / 255.

def init_params():
    w1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    w2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return w1, b1, w2, b2

def one_hot(y):
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def ReLU(x):
    return np.maximum(0,x)

def deriv_ReLU(x):
    return x > 0

def softmax(x):
    ans =  np.exp(x) / sum(np.exp(x)) #use exponential here to get rid of negative probabilities and 
    return ans #emphasis larger values to allow the algorithm to pick numbers with more certainty

def forward_propagation(w1,b1,w2,b2,X):
    nActiv1 = w1.dot(X) + b1
    activ1 = ReLU(nActiv1)
    nActiv2 = w2.dot(activ1) + b2
    activ2 = softmax(nActiv2)
    return nActiv1, activ1, nActiv2, activ2

def back_prop(nActiv1, activ1, nActiv2, activ2, w2, x, y):
    one_hot_y = one_hot(y)
    dnActiv2 = activ2 - one_hot_y
    dw2 = 1 / m * dnActiv2.dot(activ1.T)
    db2 = 1 / m * np.sum(dnActiv2)
    dnActiv1 = w2.T.dot(dnActiv2) * deriv_ReLU(nActiv1)
    dw1 = 1 / m * dnActiv1.dot(x.T)
    db1 = 1 / m * np.sum(dnActiv1)
    return dw1, db1, dw2, db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - (dw1*alpha)
    b1 = b1 - (db1*alpha)
    w2 = w2 - (dw2*alpha)
    b2 = b2 - (db2*alpha)
    return w1,b1,w2,b2

def gradient_descent(x,y,iterations, alpha):
    w1,b1,w2,b2 = init_params()
    for i in range(iterations):
        nActiv1, activ1, nActiv2, activ2 = forward_propagation(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2 = back_prop(nActiv1, activ1,nActiv2, activ2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if i % 10 == 0:
            print("Iteration number:", i)
            print("Accuracy:" , get_accuracy(np.argmax(activ2,0), y) )
    return w1,b1,w2,b2

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

w1, b1, w2, b2 = gradient_descent(X_train,Y_train, 500, 0.1)