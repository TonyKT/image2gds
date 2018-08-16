from matplotlib import pyplot as plt
import cv2
import numpy as np
import gdspy
import argparse

def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

def image2corners(args):
    filename = args.input_image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.01)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    
    # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # points = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)[1:,:]
    points=centroids[1:,:]
    # print(points)
    ind=np.lexsort((points[:,0],points[:,1]))
    corners=points[ind]
    print(corners)
    edges=auto_canny(img)
    ans = []
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y, x] != 0:
                ans = ans + [[x, y]]
    ans = np.array(ans)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    # plt.ylim(max(corners[:,1]),0)
    # plt.xlim(0,max(corners[:,0])+400)
    plt.ylim(150,0)
    plt.xlim(0,600)
    plt.scatter(ans[:,0],ans[:,1],marker='.')
    plt.scatter(corners[:,0],corners[:,1],marker='o',color='r')
    for i, xy0 in enumerate(corners):
        xy=tuple(xy0)
        ax.annotate('%d'%i,xy=xy,size=10,color='b')
    plt.savefig('foo.png')
    return corners

def corners2gds(corners,args):
    poly_cell=gdspy.Cell('tmp')
    polyidx=[None]*3 
    polyidx[0]=[0, 1, 2, 5, 4, 3, 8, 7, 6]
    polyidx[1]=[18,20,19,17,28,27,12,11,26,30,10,14,13,9,29,25,22,21,24,23,16,15,39,40,29,31,41,42,37,38,43,44,32,44,30,45,46,33,34,51,49,50,52,35,36,47,48]
    polyidx[2]=[53,54,55,58,56,57,59,60,58,59,60,61,62]
    for i, idx in enumerate(polyidx):
        polyvertice=idx2xy(corners,idx)
        poly=gdspy.Polygon(polyvertice,1)
        poly_cell.add(poly)
    if args.out_file is not None: 
        gdspy.write_gds(args.out_file,unit=1.0e-6,precision=1.0e-9)

def idx2xy(corners,idx):
    cornerList=[]
    for i in idx:
        cornerList.append(corners[i])
    return cornerList
    
parser=argparse.ArgumentParser(description='This script is used to extract corner points from image')
parser.add_argument('-i',dest='input_image',required=True,help='input image file',type=str)
parser.add_argument('-o',dest='out_file',help='output file',type=str)
args=parser.parse_args()

corners=image2corners(args)
corners2gds(corners,args)
