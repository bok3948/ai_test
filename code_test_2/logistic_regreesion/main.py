import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data preprocessing
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"

data = []
for root in [train_root, test_root]:
    #feature enginerring

    df = pd.read_csv(root)
    feat_list = list(df.columns)
    df.info()
    if root == train_root:
        Y = np.array(df["Survived"], np.dtype("float64"))
        df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    else:
        df_Y = pd.read_csv(test_root.replace("test.csv", "test_Y.csv"))
        Y = np.array(df_Y["Survived"], np.dtype("float64"))
        df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    df["Age"].fillna(df["Age"].mean(), inplace=True)

    most_freq = df["Embarked"].mode()[0]
    df["Embarked"].fillna(most_freq, inplace=True)

    label_encoder = LabelEncoder()

    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

    X = np.array(df.values, np.dtype("float64"))
    X = X.T
    Y = Y.reshape(1, -1)
    data.append((X, Y))
    print(X.shape, Y.shape)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def model(X, Y, lr, iterations):

    m = X.shape[1]
    n = X.shape[0]

    W = np.zeros((n,1))
    B = 0

    cost_list = []

    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)

        #cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        #gradient descent

        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A-Y)

        W = W - lr*dW.T
        B = B - lr*dB

        cost_list.append(cost)

        if i %(iterations//5) == 0:
            print(f"[{i}/{iterations}] loss: {cost:.4f}")

    return W, B, cost_list
    
iterations = 100000
learning_rate = 0.0015
W, B, cost_list = model(data[0][0], data[0][1], learning_rate, iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()

def metric(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    A = A > 0.5

    TP = np.sum((A == 1) & (Y == 1))
    TN = np.sum((A == 0) & (Y == 0))
    FP = np.sum((A == 1) & (Y == 0))
    FN = np.sum((A == 0) & (Y == 1))

    precision = TP / (TP + FP) #오알람률?
    recall = TP /(TP + FN)  # 놓칠확률
    f1_score = 2*(precision*recall) / (precision+recall)

    accuracy = (TP+TN) / (TP + TN + FP + FN) * 100 # negative 잘 맟추는 것도 포함

    print(f"f1_score {f1_score:.4f} Accuracy: {accuracy:.2f} %")

metric(data[1][0], data[1][1], W, B)