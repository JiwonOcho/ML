import pandas as pd
import numpy as np
import os
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data = pd.read_csv('gpascore.csv')

data = data.dropna()
print(data.isnull().sum())
print(data['gpa'].min())
print(data['gpa'].count())

y_data = data['admit'].values
# print(y_data)
x_data = []

#iterrows는 데이터프레임을 가로 한 줄씩 출력해줌
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
]) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_data),np.array(y_data), epochs=1000)

#예측
predi = model.predict([[750,3.70,3], [400,2.2,1]])
print(predi)