import sklearn
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score

#data preprocessing
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"

roots = [train_root, test_root]
datas = []
for root in roots:
    if root == train_root:
        df = pd.read_csv(root)
        Y = np.array(df["Survived"])
        
        #remove unnecessary feats
        df.drop(['PassengerId','Survived','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    else:
        df = pd.read_csv(root)
        df_y = pd.read_csv(root.replace("test.csv","test_Y.csv"))
        Y = np.array(df_y["Survived"])
        test_ids = list(df['PassengerId'])
        df.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    #handle Nan
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    most_common = df["Embarked"].mode()[0]
    df["Embarked"].fillna(most_common, inplace=True)
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)
    print(df.isna().sum())

    #category 
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])
    #label_encoder = LabelEncoder()
    #df["Sex"] = label_encoder.fit_transform(df["Sex"])
    #print(label_encoder.classes_)
    #df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
    #print(label_encoder.classes_)
    df.info()

    X = np.array(df)
    Y = Y.reshape(-1, )
    print(X.shape, Y.shape)
    datas.append((X, Y))

logistic_regressor = LogisticRegression(max_iter=100)
logistic_regressor.fit(*datas[0])
#logistic_regressor.fit(*datas[0])
#rf = RandomForestClassifier(n_estimators=100)
#svm = SVC(probability=True)
#ensemble_model = VotingClassifier(estimators=[("lr",  logistic_regressor), ("svc", svm), ("rf", rf)], voting="soft")
#ensemble_model.fit(*datas[0])



score = cross_val_score(logistic_regressor, datas[0][0], datas[0][1], cv=5, scoring=make_scorer(f1_score))

A = logistic_regressor.predict(datas[1][0])
test_score = f1_score(datas[1][1], A)
acc = accuracy_score(datas[1][1], A)
print(test_score, acc)

#save
output = pd.DataFrame({"PassengerId": test_ids, "Survived": A})
output.to_csv("/mnt/d/data/tabular/predictions.csv", index=False)

   
        