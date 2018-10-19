import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images.jpg')

#canny_edge Detection with minVal=100,maxVal=100
canny_edges1 = cv2.Canny(img,100,100)

#canny_edge Detection with minVal=100,maxVal=200
canny_edges2 = cv2.Canny(img,100,200)


#canny_edge Detection with minVal=100,maxVal=300
canny_edges3 = cv2.Canny(img,100,300)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(canny_edges1,cmap = 'gray')
plt.title('EdgeImage:minVal=100,maxVal=100'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(canny_edges2,cmap = 'gray')
plt.title('EdgeImage:minVal=100,maxVal=200'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(canny_edges3,cmap = 'gray')
plt.title('EdgeImage:minVal=100,maxVal=300'), plt.xticks([]), plt.yticks([])

plt.show()