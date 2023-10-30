import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import pandas as pd

train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"

data = []
for root in [train_root, test_root]:
    df = pd.read_csv(root)
    print(df.columns)
    if root == train_root:
        Y = np.array(df['Survived'])
        df.drop(['PassengerId', 'Survived', 'Name', 'Cabin', "Ticket"], axis=1, inplace=True)

    else:
        df_y = pd.read_csv(root.replace("test.csv", "test_y.csv"))
        Y = np.array(df_y["Survived"])
        df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    
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
    print(X.shape, Y.shape, X.dtype, Y.dtype)

def sigmoid(x):
    return 1/(1+np.exp(-x))


def model(X, Y, learning_rate, iterations):
    n, m = X.shape
    W = np.zeros((n, 1))
    B = 0
    cost_list = []
    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)

        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        #gradient descent
        dW = (1/m)*np.dot(A - Y, X.T)
        dB = (1/m)*np.sum(A-Y)

        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        cost_list.append(cost)
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

    TP = np.sum((A==1) & (Y==1))
    TN = np.sum((A==0) & (Y==0))
    FP = np.sum((A==1) & (Y==0))
    FN = np.sum((A==0) & (Y==1))

    precision = TP/ (TP+FP)
    recall = TP / (TP+FN)
    f1_score =  2*(precision*recall) / (precision+recall)
    accuracy = (TP+TN) / (TP+TN+FN+FP) *100

    print(f"F1 score: {f1_score:.4f} Accuracy: {accuracy:.4f}%")

metric(data[1][0], data[1][1], W, B)



