import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

#data preprocessing
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"

data = []
for root in [train_root, test_root]:
    df = pd.read_csv(root)
    if root == train_root:
        Y = np.array(df["Survived"])
        df.drop(["PassengerId", "Name", "Cabin", "Survived", "Ticket"], axis=1, inplace=True)
    else:
        df_y = pd.read_csv(root.replace("test.csv", "test_y.csv"))
        Y = np.array(df_y["Survived"])
        df.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1, inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    most = df["Embarked"].mode()[0]
    df["Embarked"].fillna(most, inplace=True)

    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

    X = np.array(df)
    X = X.T
    Y = Y.reshape(1, -1)
    data.append((X, Y))

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def model(X, Y, learning_rate, iterations):
    N, M = X.shape

    W = np.zeros((N, 1))
    B = 0
    cost_list = []

    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)

        cost = -(1/M)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        #gradient descent
        dW = (1/M)*np.dot(A-Y, X.T)
        dB = (1/M)*np.sum(A-Y)

        W = W - learning_rate*dW.T
        B = B - learning_rate*dB

        cost_list.append(cost)

    return W, B, cost_list

learning_rate = 0.0015
iterations = 100000
W, B, cost_list = model(data[0][0], data[0][1], learning_rate, iterations)

def metric(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    A = A > 0.5
    
    TP = np.sum((A==1) & (Y==1))
    FP = np.sum((A==1) & (Y==0))
    TN = np.sum((A==0) & (Y==0))
    FN = np.sum((A==0) & (Y==1))

    precision = TP / (TP+FP)
    recall = TP / (TP + FN)
    f1_score = 2*(precision*recall)/(precision+recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"F1_score {f1_score:.4f} Accuracy: {accuracy:.2f}%")

metric(data[1][0], data[1][1], W, B)

