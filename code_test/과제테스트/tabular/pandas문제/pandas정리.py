import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/jennybc/gapminder/master/inst/extdata/gapminder.tsv', sep='\t')

#문제1. [Quiz] pop 평균보다 인구가 높은 국가의 1970년대 데이터를 필터링하고,continent, country, pop, gdpPercap 열만 출력하라.

df.info()

print(df['year'].unique())
df1 = df[(df["pop"] >= df["pop"].mean()) & (df["year"].isin([1972, 1977]))]
df2 = df1[['continent','country', 'pop', 'gdpPercap']]

#문제2. 문제1에서 추출한 데이터프레임에서 한국, 중국, 일본의 데이터를 추출하고, 국가별로 그룹화해서 인구수의 평균을 구하라.

east_asia = ['Korea, Rep.','Japan','China']
df2 = df2[df2['country'].isin(east_asia)]
df3 = df2.groupby(["country"])["pop"].mean()
print(df3)

#문제 3. 1970년 대륙별 인구수 %로 구하여 df로 정리하라 단, %가 높은 순으로 정렬할것

# 첫 번째 데이터프레임
data1 = {
    '국가': ['한국', '한국', '한국', '중국', '중국', '중국', '일본', '일본', '일본', '미국', '미국', '미국'],
    '연도': [1970, 1975, 1980, 1970, 1975, 1980, 1970, 1975, 1980, 1970, 1975, 1980],
    '인구': [30000, 35000, 40000, 800000, 850000, 900000, 100000, 105000, 110000, 200000, 250000, 300000]
}
df1 = pd.DataFrame(data1)

# 두 번째 데이터프레임
data2 = {
    '국가': ['한국', '중국', '일본', '미국'],
    '대륙': ['아시아', '아시아', '아시아', '북아메리카']
}
df2 = pd.DataFrame(data2)

df = pd.merge(df1, df2, how='inner', on=['국가']).reset_index(drop=True)
df3 = df[df['연도'].isin([1970])]
print(df3.head())
df4 = df3.groupby(['대륙'])['인구'].sum()
print(df4)

# 전체 인구수로 나눠서 비율 계산
population_percentage = (df4 / df4.sum()) * 100
print(population_percentage)
# 결과를 데이터프레임으로 변환
result_df = population_percentage.reset_index()
result_df.columns = ['대륙', '인구수 비율 (%)']

# 인구수 비율이 높은 순으로 정렬
result_df_sorted = result_df.sort_values(by='인구수 비율 (%)', ascending=False).reset_index(drop=True)
