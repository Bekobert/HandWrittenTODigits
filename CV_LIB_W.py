import numpy as np
import cv2
import tensorflow as tf
from DrawingArea import Board 


#from HTD import HtoD_model, x_train

X_data = []
IMG_SIZE = 28

#img = cv2.imread('14.png', cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img = Board.OUTPUT
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
X_data.append(img)

X_data = np.array(X_data)
X_data = X_data.astype('float32') / 255



#print(x_train[0].shape)

cv2.imshow('image', img)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
#cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 



import matplotlib.pyplot as plt

plt.imshow(X_data[0])
plt.show()

HtoD_model = tf.keras.models.load_model('handToD.model')

pred = HtoD_model.predict([X_data])

print(np.argmax(pred[0]))