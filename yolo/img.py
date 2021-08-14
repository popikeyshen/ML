


import cv2
import numpy as np


frame = cv2.imread("1.jpg")
print(frame.shape)

frame = frame.T.reshape(-1)
print(frame.shape)

frame = np.insert(frame, 0, 123)


print(frame.shape)
print(frame)
