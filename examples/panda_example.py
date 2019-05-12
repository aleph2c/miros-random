# source:
# https://www.youtube.com/watch?v=e60ItwlZTKM

import numpy as np
import pandas as pd

def header(msg):
  print('-' * 71)
  print('[' + msg + ']')

# 1. hard coded data
header("1. load hard-coded data into df")
df = pd.DataFrame(
  [ 
    ['Jan', 58, 42, 74, 22, 2.95],
    ['Feb', 61, 42, 74, 22, 2.95],
    ['Mar', 59, 42, 74, 22, 2.95],
    ['Apr', 58, 42, 74, 22, 2.95],
    ['May', 58, 42, 74, 22, 2.95],
    ['Jun', 58, 42, 74, 22, 2.95],
    ['Jul', 58, 42, 74, 22, 2.95],
    ['Aug', 58, 42, 74, 22, 2.95],
    ['Sep', 58, 42, 74, 22, 2.95],
    ['Oct', 58, 42, 74, 22, 2.95],
    ['Nov', 58, 42, 74, 22, 2.95],
    ['Dec', 58, 42, 74, 22, 2.95]
  ],
  index = [0,1,2,3,4,5,6,7,8,9,10,11],
  columns = ['month', 'avg_high', 'avg_low', 'record_high', 'record_low', 'avg_precipitation']
)
print(df)

# 2. pull in csv file
header("2. read text from csv file")
df = pd.read_csv("Fremont_weather.txt")
print(df)

# 3. print the first 5 or last 3 rows of a df
header("3. df.head()")
print(df.head())
header("3 dt.tail(3)")
print(df.tail(3))

# 4. get data types, index, columns, values
header("4. df.types")
print(df.dtypes)  # attribute

header("4. df.index")
print(df.index)  # attribute

header("4. df.columns")
print(df.columns)  # attribute

header("4. df.values")
print(df.values)  # attribute

# 5. statistical summary of this data
header('5. df.describe()')
print(df.describe())  # method

# 6. sort records by any column
header("6 df.sort_values('record_high', ascending=False)")
print(df.sort_values('record_high', ascending=False))

# 7. slicing records
header("7. slicing -- df.avg_low")
print(df.avg_low)

header("7. slicing -- df[2:4]")
print(df[2:4])  # row 2 to 3

header("7. slicing -- df[['avg_low', 'avg_high']]")
print(df[['avg_low','avg_high']])  # get two rows

header("7. slicing -- df.loc[:, ['avg_low', 'avg_high']]")
print(df.loc[:, ['avg_low', 'avg_high']])  # multiple columns: df.loc[from_row:to_row, ['column1', 'column2']]

header("7. slicing scalar value -- df.loc[9, ['avg_precipitation']]")
print(df.loc[9,['avg_precipitation']])

header("7. df.iloc[3:5, [0,3]]")
print(df.iloc[3:5,[0,3]])  # rows 3 to 5, columns 0 and 3

# 8. filtering
header("8. df[df.avg_precipitation > 1.0]")  # filter on column values
print(df[df.avg_precipitation > 1.0])

header("8. df[df['month'].isin['Jun', 'Jul', 'Aug']]")
print(df[df['month'].isin(['Jun', 'Jul', 'Aug'])])

# 9. assignment -- very similar to slicing
header("9. df.loc[9, ['avg_precipitation']] = 101.3")
df.loc[9, ['avg_precipitation']] = 101.3
print(df.iloc[9:11])

header("9. df.loc[9, ['avg_precipitation']] = np.nan")
df.loc[9, ['avg_precipitation']] = np.nan
print(df.iloc[9:11])

header("9. df.loc[:, 'avg_low'] = np.array([5] * len(df))")
df.loc[:, 'avg_low'] = np.array([5] * len(df))
print(df.head())

header("9. df['avg_day'] = (df.avg_low + df.avg_high) / 2")
df['avg_day'] = (df.avg_low + df.avg_high) / 2
print(df.head())

# 10. renaming columns
header("10. df.rename(columns = {'avg_precipitation': 'avg_rain'}, inplace=True")
df.rename(columns = {'avg_precipitation': 'avg_rain'}, inplace=True)
print(df.head())

header("10. df.columns = ['month', 'av_hi', 'av_lo', 'rec_hi', 'rec_lo', 'av_rain', 'av_day']")
df.columns = ['month', 'av_hi', 'av_lo', 'rec_hi', 'rec_lo', 'av_rain', 'av_day']
print(df.head())

# 11. iterate a df (not needed)
header("11. iterate rows of df with a for loop")
for index, row in df.iterrows():
	print(index, row["month"], row["av_hi"])

# 12. write to csv
df.to_csv("foo.csv")

