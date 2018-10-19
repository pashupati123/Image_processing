import sys, cv2, numpy as np
from matplotlib import pyplot as plt

def equalizeHistogram(I):
	I = cv2.equalizeHist(I)
	return I

def transform(x, y, M):
    xd = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    yd = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    c = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    xd /= c
    yd /= c
    return xd, yd

def multiplyMatrix(A, B):
    C = np.zeros((A.shape[0], B.shape[1]), dtype = 'float32')
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

def stitch(img1, img2, M):

    h1 = img1.shape[0]
    print h1
    w1 = img1.shape[1]
    h2 = img2.shape[0]
    w2 = img2.shape[1]
    print w2

    dim1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype = 'float32')
    dim2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype = 'float32')

    for ele in dim2:
        ele[0], ele[1] = transform(ele[0], ele[1], M)

    dim = np.concatenate((dim1, dim2), axis = 0)

    [xMin, yMin] = np.int32(dim.min(axis=0).reshape(-1) - 0.5)
    [xMax, yMax] = np.int32(dim.max(axis=0).reshape(-1) + 0.5)

    translationMatrix = np.array([[1, 0, -xMin], [0, 1, -yMin], [0, 0, 1]])

    finalMatrix = multiplyMatrix(translationMatrix, M);

    res = np.zeros((yMax - yMin + 1, xMax - xMin + 1), dtype = 'float32')

    for i in range(h2):
        for j in range(w2):
            y, x = transform(j, i, finalMatrix)
            x = int(x)
            y = int(y)
            res[x, y] = img2[i, j]
            res[x - 1, y] = img2[i, j]
            res[x + 1, y] = img2[i, j]
            res[x, y - 1] = img2[i, j]
            res[x, y + 1] = img2[i, j]
            res[x - 1, y - 1] = img2[i, j]
            res[x + 1, y + 1] = img2[i, j]
            res[x - 1, y + 1] = img2[i, j]
            res[x + 1, y - 1] = img2[i, j]

    resCopy = np.copy(res)

    for j in range(w1):
        for i in range(h1):
            y, x = transform(j, i, translationMatrix)
            x = int(x)
            y = int(y)
            res[x, y] = img1[i, j]
    #print('Stitch Successful!')
    return res, resCopy

if __name__ == '__main__':

    img1 = cv2.imread('11.png', 0)
    img2 = cv2.imread('22.png', 0)
    I1 = equalizeHistogram(img1)
    I2 = equalizeHistogram(img2)

    #obj1 = cd.CornerDetection(I1, 3)
    #pts1 = obj1.detector(1)
    #print(pts1)
    #obj2 = cd.CornerDetection(I2, 3)
    #pts2 = obj2.detector(2)
    #print(pts2)

   # pts1 = np.float32([[285,53], [267,124], [233,111], [203, 214], [280, 173], [250, 149], [319, 135],
                       # [332,116], [366, 112], [384, 143], [384, 208], [426, 184], [467, 194], [294, 280],
                        #[332, 332], [291, 407], [260, 444], [226, 540], [289, 525], [210, 451]])

    #pts2 = np.float32([[119,44], [95, 110], [59, 88], [14, 194], [104, 164], [66, 240], [146, 133],
                        #[163, 120], [197, 121], [211, 154], [209, 216], [247, 200], [280, 214], [112, 278],
                        #[148, 336], [99, 410], [61, 451], [11, 556], [84, 534], [2, 459]])

    pts1 = np.float32([[169,55], [169,71], [169,84], [153,87], [156,103], [161,124], [169,132],
                       [129,173], [207,253], [119,237], [175,242], [175,293], [163,220], [210,331],
                       [218,365], [258,373], [247,403], [127,430], [102,392],[169,55]])

    pts2 = np.float32([[105,9], [95,20], [79,22], [84,56], [93,65], [75,72], [62,83],
                       [75,109], [93,125], [111,138], [123,131], [82,158], [70,147], [116,156],
                       [82,179], [111,190], [98,215], [93,233], [70,243],[70,147]])


    MT = cv2.findHomography(pts2, pts1, 0)
    M = MT[0]

    resultImage, resultPerspective = stitch(img1, img2, M)
    cv2.imwrite('resultPerspective.jpg', resultPerspective)
    cv2.imwrite('result.jpg', resultImage)
