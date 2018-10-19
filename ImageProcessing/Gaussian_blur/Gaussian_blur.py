import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images.jpg')

#when Sigma=0
blurred_result=cv2.GaussianBlur(img,(5,5),0)

#when Sigma=1
blurred1_result=cv2.GaussianBlur(img,(5,5),1)

#when Sigma=2
blurred2_result=cv2.GaussianBlur(img,(5,5),2)


plt.subplot(221),plt.imshow(img),plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blurred_result),plt.title('Blurred image:Sigma=0')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(blurred1_result),plt.title('Blurred image:Sigma=1')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(blurred2_result),plt.title('Blurred image:Sigma=2')
plt.xticks([]), plt.yticks([])
plt.show()