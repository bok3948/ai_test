import sklearn
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

root = "/mnt/d/data/tabular/housing.csv"
df = pd.read_csv(root, header=None)
data = [list(map(float, row[0].split())) for index, row in df.iterrows()]
data = np.array(data)
Y = data[:, -1]
X = data[:, :-1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 피처 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred_sklearn = regressor.predict(X_test)
MSE_sklearn = mean_squared_error(Y_test, Y_pred_sklearn)
print(f"sklearn linear model: {MSE_sklearn}")



from sklearn.tree import DecisionTreeRegressor

# 결정 트리 회귀 모델 초기화 및 학습
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, Y_train)

# 예측 및 성능 평가
Y_pred_tree = tree_regressor.predict(X_test)
MSE_tree = mean_squared_error(Y_test, Y_pred_tree)
RMSE_tree = np.sqrt(MSE_tree)
R2_score_tree = tree_regressor.score(X_test, Y_test)

print(f"MSE_tree {MSE_tree}")



def model(X, Y, learning_rate, iterations):
    X = X.T
    Y = Y.reshape(1, -1)
    n, m = X.shape
    W = np.zeros((n, 1))
    B = 0
    cost_list = []
    for i in range(iterations):
        A = np.dot(W.T, X) + B

        cost = (1/m)*np.sum((A - Y)**2)
        dW = (1/m)* np.dot((A - Y), X.T)
        dB =  (1/m)*np.sum(A - Y)

        W = W - learning_rate*dW.T
        B = B - learning_rate*dB

        cost_list.append(cost)

    return W, B, cost_list

iterations = 2000
learning_rate = 0.005
W, B, cost_list = model(X_train, Y_train, learning_rate, iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()

def metric(X, Y, W, B):
    X = X.T
    Y = Y.reshape(1, -1)
    n, m = X.shape
    A = np.dot(W.T, X) + B

    MSE = (1/m)*np.sum((A - Y)**2)

    Rpow2_score =  1 - (np.sum((A - Y)**2)/np.sum((Y - np.mean(Y))**2))

    print(f"MSE: {MSE:.4f} R^2 score: {Rpow2_score}")

metric(X_test, Y_test, W, B)
