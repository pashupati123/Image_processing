
import cv2
import numpy as np
from matplotlib import pyplot as plt

src_img = cv2.imread('2.jpg',0)
src_img = np.float32(src_img)
corner_detection = cv2.cornerHarris(src_img,2,3,0.20)
print corner_detection

plt.subplot(2,1,1), plt.imshow(src_img)
plt.title('Original mage'), plt.xticks([]), plt.yticks([])

src_img2 = cv2.imread('2.jpg')
corners2_detection = cv2.dilate(corner_detection, None, iterations=3)
src_img2[corners2_detection>0.01*corners2_detection.max()] = [255,0,0]

plt.subplot(2,1,2),plt.imshow(src_img2,cmap = 'gray')
plt.title('Corner Detection'), plt.xticks([]), plt.yticks([])

plt.show()