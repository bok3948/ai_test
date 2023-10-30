import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#preprocessing boston housing
root = "/mnt/d/data/tabular/housing.csv"
df = pd.read_csv(root, header=None,  delimiter=r"\s+")
df.info()
#tem = [list(map(float, row[0].split())) for index, row in df.iterrows()]
data = np.array(df)
X = data[:, :-1]
Y = data[:, -1]
Y = Y.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.lr = learning_rate
        self.iters = iterations
        self.W = None
        self.B = None

    def fit(self, X, Y):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.B = 0
        for i in range(self.iters):
            A = np.dot(X, self.W) + self.B
            cost = np.mean((A - Y)**2)
        
            dW = (1/m)*np.dot(X.T, A-Y)
            dB = (1/m)*np.sum(A-Y)

            self.W = self.W - self.lr*dW
            self.B = self.B - self.lr*dB

    def predict(self, X):
        A = np.dot(X, self.W) + self.B
        return A
    
reg = LinearRegression(0.015, 10000)
reg.fit(X_train, Y_train)
A = reg.predict(X_test)

def mse(A, Y_test):
    return np.mean((A-Y_test)**2)

mse_score = mse(A, Y_test)
print(f"MSE: {mse_score:.4f}")