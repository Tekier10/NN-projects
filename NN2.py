import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

df_train = pd.read_csv(r"C:\Users\micha\Downloads\neural-nets-master\neural-nets-master\examples\quadratic\data\train.csv")
df_test = pd.read_csv(r"C:\Users\micha\Downloads\neural-nets-master\neural-nets-master\examples\quadratic\data\test.csv")
x_test = np.column_stack((df_test.x.values, df_test.y.values))
df_train = np.array(df_train)
np.random.shuffle(df_train)
df_train = pd.DataFrame(df_train, columns = ['x','y','color'])
x = np.column_stack((df_train.x.values, df_train.y.values))
print(df_train.shape)

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(2,), activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid")])

model.compile(optimizer='adam',
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

model.fit(x, df_train.color.values, batch_size=16, epochs=5)
print("EVALUATION")
model.evaluate(x_test, df_test.color.values)