import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error

root = "/mnt/d/data/tabular/housing.csv"
df = pd.read_csv(root)
#df.info()
print(df.columns)
#for i in range(len(df.columns)):
#    print(df.iloc[:5, i])
Y = df["median_house_value"]
df.drop(["median_house_value"], axis=1, inplace=True)
df.fillna(int(df["total_bedrooms"].mean()), inplace=True)
df = pd.get_dummies(df, ["ocean_proximity"])
X = df.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
A = regressor.predict(X_test)

score = mean_squared_error(Y_test, A)
print(score)

