import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser=argparse.ArgumentParser(description='This script is used to extract corner points from image')
parser.add_argument('-i',dest='input_image',required=True,help='input image file',type=str)
args=parser.parse_args()
# im = cv2.imread(args.input_image)
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# edge = cv2.Canny(thresh, 100, 200)
# (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im = cv2.imread(args.input_image)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

total = 0
for c in cnts:
    epsilon = 0.08 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    cv2.drawContours(im, [approx], -1, (0, 255, 0), 4)
    total += 1

print "I found {0} RET in that image".format(total)
cv2.drawContours(im2, contours, -1, (0,255,0), 3)
cv2.imwrite('contour.png',im2)
# cv2.imshow("Output", img)
# cv2.waitKey(0)
# exit()
