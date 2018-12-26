import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#mnist = tf.keras.datasets.mnist

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

labels = np.array(train['label'].tolist()).astype(np.uint8)
features = train.drop(['label'], axis=1)
features = np.array(features).astype(np.uint8)
features = features.reshape(42000, 28, 28)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#plt.imshow(x_train[0], cmap=plt.cm.binary)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
loss, accuracy = model.evaluate(x_test, y_test)

prediction = model.predict(x_test)
norm_pred = np.argmax(prediction)

test_pred = model.predict(test)

submission = pd.DataFrame()
label = []
id = []
for i in range(1, 28001):
    id.append(i)


submission['ImageId'] = id
submission['Label'] = label

submission.to_csv('Digit_Recognizer.csv', index=False)