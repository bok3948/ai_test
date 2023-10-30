import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("/mnt/d/data/tabular/boston_housing.csv", header=None)
df = [list(map(float, row[0].split()))for idx, row in df.iterrows()]
df = np.array(df)
df = pd.DataFrame(df, columns=column_names)
df.info()
for i in range(len(df.columns)):
    print(len(set(list(df.iloc[:, i]))))



