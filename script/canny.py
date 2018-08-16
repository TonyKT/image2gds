import cv2
import numpy as np
from matplotlib import pyplot as plt
# from shapely import geometry
import gdspy

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	
	# return the edged image
	return edged

img = cv2.imread('Ferrari.png',0)
# edges = cv2.Canny(img,100,200)
edges=auto_canny(img)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
# print('edges'+edges.shape)
# print edges[0:10,0:10]
ans = []
for y in range(0,edges.shape[0]):
    for x in range(0, edges.shape[1]):
        if edges[y, x] != 0:
            # ans = ans + [[x, y]]
            ans.append((x,y))
ans = np.array(ans)

#need one function to get the polygon end point

print(ans.shape)
print('ans0')
print(ans[0:10,0])
print(ans[-10:-1,0])
print('ans1')
print(ans[0:10,1])
print(ans[-10:-1,1])
# for i,j in ans:
#     print i,j

# poly=geometry.Polygon(ans)
# print(poly.wkt)

# x,y=poly.exterior.xy
# print('x',len(x))
# print(x[0:10])
# print(x[-10:-1])
# print('y',len(y))
# print(y[0:10])
# print(y[-10:-1])
#
# fig=plt.figure(1,figsize=(5,5),dpi=90)
# ax=fig.add_subplot(111)
# ax.plot(x,y,color='#6699cc',alpha=0.7,linewidth=3,solid_capstyle='round',zorder=2)
# ax.set_title('Polygon')
# plt.show()


poly_cell=gdspy.Cell('tmp')
points=ans
# points=[(0,0),(2,2),(2,6),(-6,6),(-6,-6),(-4,-4),(-4,4),(0,4)]
poly1=gdspy.Polygon(points,1)
poly_cell.add(poly1)

gdspy.write_gds('test.gds',unit=1.0e-6,precision=1.0e-9)
# gdsii=gdspy.GdsLibrary()
# gdsii.read_gds('test.gds')
# gdsii.extract('tmp')

# gdspy.LayoutViewer()
