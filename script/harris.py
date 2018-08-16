import cv2
import numpy as np
# from matplotlib import pyplot as plt
import kMeans

img = cv2.imread('Ferrari2.png',0)
img = np.float32(img)
corners = cv2.cornerHarris(img,2,3,0.04)
# plt.subplot(2,1,1), plt.imshow(corners ,cmap = 'jet')
# plt.title('Harris Corner Detection'), plt.xticks([]), plt.yticks([])

img2 = cv2.imread('Ferrari2.png')
corners2 = cv2.dilate(corners, None, iterations=3)
crit=0.01*corners2.max()
img2[corners2 > crit] = [255,0,0]
# print 'corners'
# print corners
# print 'corners2'
# print corners2
# plt.subplot(2,1,2),plt.imshow(img2,cmap = 'gray')
# plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
#
# plt.show()

target = []
for y in range(0, corners2.shape[1]):
    for x in range(0, corners2.shape[0]):   
        if corners2[x, y] >crit:
            target = target + [[x, y]]
target = np.array(target)
print target.shape

myCentroids, clustAssing = kMeans.kMeans(target,36)
print myCentroids
