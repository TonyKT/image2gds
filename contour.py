import numpy as np
import cv2
import argparse

def getContour(args):
    im = cv2.imread(args.input_image)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print contours
    cv2.drawContours(im2, contours, -1, (0,255,0), 3)
    cv2.imwrite('contour.png',im2)


parser=argparse.ArgumentParser(description='This script is used to extract corner points from image')
parser.add_argument('-i',dest='input_image',required=True,help='input image file',type=str)
args=parser.parse_args()
getContour(args)
