import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense



model = tf.keras.models.load_model('model.h5')

df1 = pd.read_csv('features1.csv')
df = df1.drop(['url'],axis=1).copy()

print(df.head())
x = df.drop('label',axis=1)
y = df['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.50,random_state=42)

print(x_test.head())

y_pred = model.predict(x)
classes_y=np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y,classes_y)
from sklearn.metrics import accuracy_score
print(accuracy_score(y, classes_y))