import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/jennybc/gapminder/master/inst/extdata/gapminder.tsv', sep='\t')
df.info()
df1 = (df["pop"] >= df["pop"].mean()) 
df1.info()
print(df['year'].unique())
df1 = df[(df["pop"] >= df["pop"].mean()) & (df["year"].isin([1972, 1977]))]
print(df1.head())

df2=df1[['continent','country', 'pop', 'gdpPercap']]
east_asia = ['Korea, Rep.','Japan','China']
df2 = df2[df2['country'].isin(east_asia)]
print(df2.head())
df3 = df2.groupby(["country"])["pop"].mean()
print(df3)














    





           


















        









