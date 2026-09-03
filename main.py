
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(Z):
    Ztransform = Z - np.max(Z)
    A = np.exp(Ztransform)/(np.sum(np.exp(Ztransform)))
    return A

def cost(Y, A):
    return (-1/len(Y))* np.sum(Y * np.log(A))

def forwardprop(X, weights, biases, layers):
    Zcache = []
    Acache = []

    A = X

    for i in range(len(weights)):
        
        Z = A @ weights[i] + biases[i]
        Zcache.append(Z)

        if i == len(weights)-1:
            A = softmax(Z)
        else:
            A = sigmoid(Z)

        Acache.append(A)

    return A, Zcache, Acache

    




# Load in the data
train_data = pd.read_csv("./input/train.csv")
test_data= pd.read_csv("./input/test.csv")
train_labels=np.array(train_data.loc[:,'label'])
train_data=np.array(train_data.loc[:,train_data.columns!='label'])

train_data = train_data.reshape(42000, 784)

layers = [784, 128, 10]

weights = []
biases = []

for i in range(len(layers)-1):
    weight = np.random.normal(loc = 0, scale = (layers[i])**(-0.5), size=[layers[i], layers[i+1]])
    weights.append(weight)
    bias = np.zeros(shape=[1,layers[i+1]])
    biases.append(bias)

Z_1 = train_data @ weights[0] + biases[0]
A_1 = sigmoid(Z_1)

Z_2 = A_1 @ weights[1] + biases[1]
A_2 = softmax(Z_2)

print(A_2.shape)