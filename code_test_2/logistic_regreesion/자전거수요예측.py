import sklearn
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest


root = "/mnt/d/data/tabular/bike"
train_root = os.path.join(root, "train.csv")
test_root = os.path.join(root, "test.csv")


datas = []
roots = [train_root, test_root]

for root in roots:

    df = pd.read_csv(root)
    if root == train_root:
        #print(len(df.columns))
        #print(df.columns)
        print(df.info())
        #print(df.isna().sum())
        Y = df["count"].to_numpy()
        df.drop(["datetime", 'casual', 'registered', 'count', "atemp", "workingday"], axis=1, inplace=True)
    
    else:
        Y = None
        test_datetime = df["datetime"]
        df.drop(["datetime",  "atemp", "workingday"], axis=1, inplace=True)

    df = pd.get_dummies(df,columns=['season', 'weather'])
    X = df.to_numpy()

    
    #outliar 제거
    if root == train_root:
        clf = IsolationForest(contamination=0.1, random_state=20)
        outliers = clf.fit_predict(X)
        mask = outliers != -1
        print(X.shape)
        X, Y = X[mask], Y[mask]
        print(X.shape)

    datas.append((X, Y))


#linear_model = RandomForestRegressor()
linear_model = GradientBoostingRegressor(n_estimators = 2000
					, learning_rate = 0.05
                                    , max_depth = 5
                                    , min_samples_leaf = 15
                                    , min_samples_split = 10
                                    , random_state = 42)
scores = cross_val_score(linear_model, datas[0][0], datas[0][1], scoring=make_scorer(mean_squared_error))
print(np.sqrt(scores))
linear_model.fit(*datas[0])
A = linear_model.predict(datas[1][0])

#save
result = pd.read_csv(os.path.join('/mnt/d/data/tabular/bike', "SampleSubmission.csv"))
submission = pd.DataFrame()
submission["datetime"] = test_datetime
A[A <= 0] = 0
submission["count"] = A

submission.to_csv("/mnt/d/data/tabular/bike/submission.csv", index=False)

 

   

