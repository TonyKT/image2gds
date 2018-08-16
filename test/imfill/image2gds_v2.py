import numpy as np
import cv2
import gdspy
import argparse
    

def auto_canny(image, sigma=0.33):
    """canny edge detection """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def getContour(args):
    img = cv2.imread(args.input_image)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image_gray = cv2.dilate(image_gray,None)
    #Otsu's Binarization
    if args.background=='white':
        image_gray= cv2.copyMakeBorder(image_gray,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        ret,dst=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    elif args.background=='black':
        image_gray= cv2.copyMakeBorder(image_gray,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        ret,dst=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        sys.exit('\tError: background color not supported. use white or black')
    # dst = cv2.dilate(dst,None)
    
    # dst=auto_canny(img)
    # dst = cv2.dilate(dst,None)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # img = cv2.dilate(dst, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=2)
    # dst = cv2.dilate(img, kernel, iterations=1)

    cv2.imwrite('contour.png',dst)
    #findContour
    dst = np.uint8(dst)
    im2, contour, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return contour

def sharpenCorner(contour,args):
    """ remove  nonHV corner """
    epsilon=3
    newContour=[]
    for poly in contour:
        poly=np.array(poly)
        nP=len(poly)
        polyA=poly[:nP-1]
        polyB=poly[1:]
        dist = [(a - b)**2 for a, b in zip(polyA, polyB)]
        if dist: 
            dist = np.sqrt(np.sum(dist,axis=1))
            for i,d in enumerate(dist):
                if d<epsilon and all(poly[i]-poly[i+1]):
                    corner=np.array([poly[i][0],poly[i+1][1]])
                    if all(poly[i+2]-corner) or all(poly[i-1]-corner):
                        corner=np.array([poly[i+1][0],poly[i][1]])
                    poly[i]=corner
                    poly[i+1]=corner
        newContour.append(poly)
    return newContour 

def contour2gds(contour0,args):
    """transform contour list to gds """    
    contour=[[ele[0] for ele in arr] for arr in contour0]
    if args.sharpen:
        contour=sharpenCorner(contour,args)
    flat_list = [item for sublist in contour for item in sublist]
    
    x=[arr[0] for arr in flat_list]
    y=[arr[1] for arr in flat_list]
    xlength=max(x)-min(x)
    ylength=max(y)-min(y)
    
    maxY=max(y)
    contour=[[[ele[0],maxY-ele[1]] for ele in arr] for arr in contour]
    

    poly_cell=gdspy.Cell('tmp')
    poly=gdspy.PolygonSet(contour,1)
    
    nX=args.nX
    nY=args.nY
    xsep=max(xlength,ylength)*args.sep
    ysep=xsep
    for i in range(nX): 
        for j in range(nY):
            xpos=(xlength+xsep)*(i+1)
            ypos=(ylength+ysep)*(j+1)
            trans=gdspy.copy(poly,xpos,ypos)
            poly_cell.add(trans)

    if args.out_file is not None: 
        gdspy.write_gds(args.out_file,unit=args.scale*1.0e-9,precision=1.0e-9)

parser=argparse.ArgumentParser(description='This script is used to extract corner points from image')
parser.add_argument('-i',dest='input_image',required=True,help='input image file',type=str)
parser.add_argument('-s',dest='corner_seq',help='image corner sequence',type=str)
parser.add_argument('-nx',dest='nX',default=1,help='replicate nx times along x axis',type=int)
parser.add_argument('-ny',dest='nY',default=1,help='replicate ny times along y axis',type=int)
parser.add_argument('-sep',dest='sep',default=0.1,help='separation ratio',type=float)
parser.add_argument('-scale',dest='scale',default=1.0,help='unit=scale nm',type=float)
parser.add_argument('-o',dest='out_file',help='output file',type=str)
parser.add_argument('-sharpen',dest='sharpen',default=False,help='sharpen corner',type=str)
parser.add_argument('-bg',dest='background',default='white',help='background color of image',type=str)
args=parser.parse_args()
contour=getContour(args)
contour2gds(contour,args)
