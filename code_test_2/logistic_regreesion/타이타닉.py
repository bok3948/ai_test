import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"

data1 = pd.read_csv(train_root)
len_train = len(data1)
data2 = pd.read_csv(test_root)
df = pd.concat((data1, data2),ignore_index=True)
df.info()
print(df.columns)
print(df.isna().sum().sort_values(ascending=False))

Y = df["Survived"][:len_train].copy()
ids = df["PassengerId"][len_train:].copy()


import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
dataset = df
dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

df =dataset

df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket", "Survived"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Age_bin'] = pd.cut(df['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
df['Fare_bin'] = pd.cut(df['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare','Average_fare','high_fare'])

df.drop(columns=["Age", "Fare"], axis=1, inplace=True)

df = pd.get_dummies(df)
feats = df.columns
print(feats)
X_train = df[:len_train].to_numpy()
X_test = df[len_train:].copy()
X_test.info()
X_test = X_test.to_numpy()

print(X_train.shape, Y.shape, X_test.shape)

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'bootstrap': bootstrap,
}
rfc = RandomForestClassifier(random_state=42)

rfc_cv = RandomizedSearchCV(estimator=rfc, scoring='accuracy', param_distributions=random_grid, n_iter=200, cv=5, verbose=1, random_state=42, n_jobs=-1)
rfc_cv.fit(X_train, Y)
rf_best_params = rfc_cv.best_params_
print(f"Best paramters: {rf_best_params})")

rfc = RandomForestClassifier(**rf_best_params)


gbc = GradientBoostingClassifier(n_estimators=100, random_state=32)
lr = LogisticRegression(max_iter=1000, random_state=32)
score1 = cross_val_score(rfc, X_train, Y)
score2 = cross_val_score(gbc, X_train, Y)
score3 = cross_val_score(lr, X_train, Y)
print(round(score1.mean()*100,2))
print(round(score2.mean()*100,2))
print(round(score3.mean()*100,2))

rfc.fit(X_train, Y)
A = rfc.predict(X_test)
print(A.shape)
importance_series = pd.Series(rfc.feature_importances_, index=feats)
importance_series = importance_series.sort_values(ascending=False)
print(importance_series)
sub = pd.DataFrame()
sub["PassengerId"] = ids
sub["Survived"] = A
sub.to_csv("/mnt/d/data/tabular/tit_submission.csv", index=False)

#choose model





