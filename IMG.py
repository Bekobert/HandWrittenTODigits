import numpy as np
import cv2
from HTD import model

X_data = []
IMG_SIZE = 50

img = cv2.imread('Three.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
X_data.append(img)

X_data = np.array(X_data)
X_data = X_data.astype('float32') / 255


model.predict(X_data[0])