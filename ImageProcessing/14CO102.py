import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_stitched_image(img1, img2, M):


    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

 
    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


    # Transformation using perspectiveTransform of Opencv
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

    # Joining images
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
    
    # Create output array
    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0,0,1]]) 

    # Manipulating(using warpPerspective) images to get the resulting image
    result_image = cv2.warpPerspective(img2, transform_array.dot(M), (x_max-x_min, y_max-y_min))
    result_image[transform_dist[1]:w1+transform_dist[1], transform_dist[0]:h1+transform_dist[0]] = img1

    return result_image

def get_sift_homography(img1, img2):
 
   # sift = cv2.xfeatures2d.SIFT_create()

    sift=cv2.SIFT()

    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1,d2, k=2)

    # Recommended Ratio is 0.8 from StackOverflow
    verify_ratio = 0.8
    verified_matches = []
    for m1,m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)  # Good Matches being added to array

    # Minimum threshold on matches
    min_matches = 20
    if len(verified_matches) > min_matches:
        
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1,1,2)
        img2_pts = np.float32(img2_pts).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print('Atleast 20 matches required')



img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('2.jpg',0)


blurred_img1 = cv2.GaussianBlur(img1,(5,5),0)
blurred_img2 = cv2.GaussianBlur(img2,(5,5),0)

M =  get_sift_homography(img1, img2)

# Stitch the images together using homography matrix
result_image = get_stitched_image(img2, img1, M)

result_image_name = 'final.jpg'
cv2.imwrite(result_image_name, result_image)

cv2.imshow ('final', result_image)
cv2.waitKey()

