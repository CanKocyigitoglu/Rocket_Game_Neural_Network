import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('less_data.csv')
print(df)

column1 = df.iloc[250000:,0].tolist()
column2 = df.iloc[250000:,1].tolist()

first_column_max_value = max(column1)
second_column_max_value = max(column2)
first_column_min_value = min(column1)
second_column_min_value = min(column2)

def normalization(first_column_max_value, first_column_min_value, second_column_max_value, second_column_min_value, first_column_value, second_column_value):
    data = []
    for i in range(len(column1)):
        X1 = (first_column_value[i] - first_column_min_value) / (first_column_max_value - first_column_min_value)
        X2 = (second_column_value[i] - second_column_min_value) / (second_column_max_value - second_column_min_value)
        data.append([X1, X2])
    return data

input_layer = normalization(first_column_max_value, first_column_min_value, second_column_max_value, second_column_min_value, column1, column2)
input_layer = np.array(input_layer)



x1points = input_layer[:,0]
x2points = input_layer[:,1]


plt.plot(x1points)
plt.plot(x2points)
plt.show()
