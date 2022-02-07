import tensorflow as tf
import numpy as np
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import matplotlib.pyplot as plt

#plt.imshow(x_train[60])
#plt.show()

#print(x_train[60]) #datas are in a range between 0 and 253! Needs a normalization.
#print("***************************************************************************")
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#print(x_train[60]) #it's better now..

#ML NN Model
model = tf.keras.models.Sequential() #A common model type, Sequential..

model.add(tf.keras.layers.Flatten()) #Flatten Input Layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #Activation Function: Rectified Linear Unit
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #Output Layer, There are 10 different possible outputs. Softmax for Output Layer.

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)

loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)

#model.save('handToD.model')
#HtoD_model = tf.keras.load_model('handToD.model')
