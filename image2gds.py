from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import gdspy
import argparse
from collections import deque


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

def image2corners(args):
    """get edges and corners from image"""
    img = cv2.imread(args.input_image)
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    points = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)[1:,:]
    # points=centroids[1:,:]
    ind=np.lexsort((points[:,0],points[:,1]))
    corners=points[ind]
    # print(corners)
    edges=auto_canny(img)
    ans = []
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y, x] != 0:
                ans = ans + [[x, y]]
    ans = np.array(ans)
    maxY=max(corners[:,1])
    corners[:,1]=maxY-corners[:,1]
    ans[:,1]=maxY-ans[:,1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.ylim(0,max(corners[:,1])*1.1)
    plt.xlim(0,max(corners[:,0])*1.1)
    # plt.ylim(150,0)
    # plt.xlim(0,600)
    plt.scatter(ans[:,0],ans[:,1],marker='.',color='gray',s=0.1)
    plt.scatter(corners[:,0],corners[:,1],marker='o',color='r',s=1)
    for i, xy0 in enumerate(corners):
        # xy=tuple(xy0+[0,0.02*maxY])
        xy=tuple(xy0)
        ax.annotate('%d'%i,xy=xy,size=6,color='b')
    plt.savefig('foo.png')
    
    return corners,ans

def autoHV(corners,ans):
    # print(ans.shape)    
    # print(corners.shape)    
    np.savetxt('corner.txt',corners) 
    # print('corners:')
    # print(corners)
    hLines=segSort(ans,'h')
    vLines=segSort(ans,'v')
    print("n-hLines,n-vLines")
    print(len(hLines),len(vLines))
    testPlot(hLines,'h')
    testPlot(vLines,'v')
    polyset=auto_connect(hLines,vLines,corners)
    poly2gds(polyset,corners,args)
    return 

def endPoint(line):
    lineArray=np.array(line)
    # print(lineArray)
    hDist=np.max(lineArray[:,0])-np.min(lineArray[:,0])
    vDist=np.max(lineArray[:,1])-np.min(lineArray[:,1])
    orient=''
    if hDist>vDist:
        orient=='h'
        ax=0
    elif hDist<vDist:
        orient=='v'
        ax=1
    else:
        # print('orient error')
        exit()
   
    sortArray=lineArray[lineArray[:,ax].argsort()]
    minP=sortArray[0]
    maxP=sortArray[-1]
    # print('ax',ax)
    # print(line)
    # print(minP)
    # print(maxP)
    return [minP, maxP]
    # return [minP,maxP,orient]

def auto_connect(hLines,vLines,corners):
    
    pairs=[]
    lines=hLines+vLines
    epsilon=2 #separate lines usually has dist larger than 2 pixel
    for line in lines:
        # print('line',line)
        # print('end',endPoint(line))
        [minP,maxP]=endPoint(line)
        if distance(minP,maxP) > epsilon:
            pairs.append([minP,maxP])
    pairs=pairsFilter(pairs)
    # print('len:',len(pairs))
    # for pair in pairs:
    #     print(pair)
    pairs=np.array(pairs)
    pairs=nearestPoint(pairs,corners)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open('pairs.txt', 'w') as f:
        f.write(np.array2string(pairs, separator=', '))
    f.close()
    #
    flat_pairs=np.array([item for sublist in pairs for item in sublist])
    # print('flat_pairs',flat_pairs)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.scatter(flat_pairs[:,0],flat_pairs[:,1],marker='o',color='gray',s=1.0)
    plt.savefig('pairs.png')
    # exit() 
    
    # pairs=np.delete(pairs,0,axis=0)
    # print('npair1',pairs.shape)
    # exit()
    # print('npair0',pairs.shape)
    polyset=list()
    polycorner=deque()
    polycorner.extend(pairs[0])
    pairs=np.delete(pairs,0,axis=0)
    setFlag=False
    # print('ployset0',len(polyset))
    while True:
        [polycorner,pairs,setFlag]=addPoint2(polycorner,pairs)
        if setFlag: 
            polyset.append(list(polycorner))
            # print('polycorner len',len(polycorner))
            # print(polycorner)
            # print('len pairs',pairs.shape[0])
            # print(pairs)
            # print('polyset len',len(polyset))
            # print(polyset)
            # exit()
            if pairs.shape[0] == 0:
                print('All polygon set found')
                break
            else:
                polycorner=deque()
                polycorner.extend(pairs[0])
                pairs=np.delete(pairs,0,axis=0)
        # print('npair:',pairs.shape)
        # print('npoly:',len(polycorner))
        # if pairs.shape[0] == 0:
        #     print('All polygon set found')
        #     break
    # print('polyset')
    # print(polyset)
    return polyset

def nearestPoint(pairs,corners):
    epsilon=4 # corner from harris(stat center), edge from canny,deviation could be large 
    for i in range(pairs.shape[0]):
        for j in range(pairs.shape[1]):
            for corner in corners:
                dist=distance(pairs[i,j], corner)
                if dist<=epsilon:    
                    # print('dist',dist,pairs[i,j],corner)
                    pairs[i,j]=corner
                    # print('pairs',pairs.dtype)
                    # print('corner',corner.dtype)
                    # print('after change',pairs[i,j])
    return pairs

def pairsFilter(pairs):
    uniqPairs=[]
    epsilon=2
    for pair in pairs:
        if uniqPairs:
            Flag=True
            for uniq in uniqPairs:
                dist=distance(uniq,pair)/len(pair)
                # print('dist',dist)
                if dist<=epsilon:
                    Flag=False
            if Flag:
                uniqPairs.append(pair)
        else:
            uniqPairs.append(pair)
    return uniqPairs

# def addPoint(polycorner,pairs,idx):
#     if idx:    
#         pairs=np.delete(pairs,idx,axis=0)
#     
#     if np.all(polycorner[0] == polycorner[-1]):
#         print('break:',polycorner)
#         return polycorner,pairs,idx
#
#     epsilon=4
#     # for i, pair in enumerate(pairs):
#     print('len pairs',pairs.shape)
#     # for i in range(pairs.shape[0]):
#     idx=[]
#     for i, pair in enumerate(pairs):
#         # pair=pairs[i]
#         print('pair:',pair)
#         print('poly:',polycorner)
#         # for j  in range(pair.shape[0]):
#         #     point=pair[j]
#         for j, point in enumerate(pair):
#             dist1=distance(point,polycorner[0])
#             dist2=distance(point,polycorner[-1])
#             print('point',point,'dist',dist1,dist2)
#             if dist1<epsilon:
#                 if j==1:
#                     polycorner.extendleft(pair[::-1])
#                 else:
#                     polycorner.extendleft(pair)
#                 # pairs=np.delete(pairs,i,axis=0)
#                 # print('head',polycorner[0])
#                 # print('tail',polycorner[-1])
#                 idx.append(i)
#                 [polycorner,pairs,idx]=addPoint(polycorner,pairs,idx)
#             elif dist2<epsilon:
#                 if j==1:
#                     polycorner.extend(pair[::-1])
#                 else:
#                     polycorner.extend(pair)
#                 # pairs=np.delete(pairs,i,axis=0)
#                 # print('head',polycorner[0])
#                 # print('tail',polycorner[-1])
#                 idx.append(i)
#                 [polycorner,pairs,idx]=addPoint(polycorner,pairs,idx)
#             else:
#                 continue

def addPoint2(polycorner,pairs):
    epsilon=4
    # print('len pairs',pairs.shape)
    setFlag=False
    idx=[]
    for i, pair in enumerate(pairs):
        # print('pair:',pair)
        # print('poly:',polycorner)
        # print('polycorner0',polycorner[0])
        # print('polycorner-1',polycorner[-1])
        endPoints=[polycorner[0],polycorner[-1]]
        for j, point in enumerate(pair):
            dist=[distance(point,endPoints[x]) for x in [0,1]]
            # print('point',point,'dist',dist)
            if dist[0]<epsilon:
                otherpoint=[pair[1-j]]
                # print('otherpoint0',otherpoint)
                # print('polycorner-1',polycorner[-1])
                idx.append(i)
                if np.all(otherpoint==polycorner[-1]):
                    setFlag=True
                    break
                else:
                    polycorner.extendleft(otherpoint[::-1])
            elif dist[1]<epsilon:
                otherpoint=[pair[1-j]]
                # print('otherpoint1',otherpoint)
                # print('polycorner0',polycorner[0])
                idx.append(i)
                if np.all(otherpoint==polycorner[0]):
                    setFlag=True
                    break
                else:
                    polycorner.extend(otherpoint)
            else:
                continue
    if idx:    
        pairs=np.delete(pairs,idx,axis=0)
    if pairs.shape[0] == 0:
        setFlag=True

    return polycorner,pairs,setFlag

def testPlot(lineData,orient):
    print("Lines")
    figname=orient+'Lines.png'
    tmpList=list()
    for i in range(len(lineData)):
        if len(lineData[i]) < 2:
            print(len(lineData[i]))
            print(lineData[i])
        # print(lineData[i])
        tmpList=tmpList+lineData[i]
    # print(len(tmpList))
    tmpArray=np.array(tmpList)
    # print(tmpArray.shape)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.scatter(tmpArray[:,0],tmpArray[:,1],marker='.',color='gray',s=0.1)
    plt.savefig(figname)
    txtname=orient+'Lines.txt'
    np.savetxt(txtname,tmpArray) 
    return 

def segSort(edges,orient):
    if orient=='h':
        ax1,ax2=0,1
    elif orient=='v':
        ax1,ax2=1,0
    else:
        print('orient error')
        exit()

    ind=np.lexsort((edges[:,ax1],edges[:,ax2]))
    ans=edges[ind]
    txtname=orient+'edge.txt'
    np.savetxt(txtname,ans) 

    y=ans[0,:].tolist()
    epsilon=2       #pixel
    orderLines=list()
    lineList=list()
    for i in range(ans.shape[0]):
        x=ans[i,:]
        dist=distance(x,y)
        if dist<=epsilon:
            # print('dist',x,dist)
            lineList.append(x.tolist()) 
        else:
            if len(lineList)>epsilon:
                orderLines.append(lineList)
            lineList=list()
        y=x
    if len(lineList)>0:
        orderLines.append(lineList)
    return orderLines


def distance(x,y):
    return np.linalg.norm(np.array(x)-np.array(y)) 

def corners2gds(corners,args):
    polyidx=[]
    with open(args.corner_seq,'r') as f:
        for line in f:
            line=line.strip().split(',')
            tmp=[int(x) for x in line]
            polyidx.append(tmp)
    poly_cell=gdspy.Cell('tmp')
    polyvertice=[]
    for i, idx in enumerate(polyidx):
        polyvertice.append(idx2xy(corners,idx))
        # poly=gdspy.Polygon(polyvertice,1)
        # poly_cell.add(poly)
    poly=gdspy.PolygonSet(polyvertice,1)
    # poly_cell.add(poly)
    xlength=max(corners[:,0])-min(corners[:,0])
    ylength=max(corners[:,1])-min(corners[:,1])
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

def poly2gds(polyvertice,corners,args):
    f=open('auto.seq','w')
    epsilon=4
    for vset in polyvertice:
        # print('vset',vset)
        seqList=[]
        for v in vset:
            for i,corner in enumerate(corners):
                dist=distance(v, corner)
                if dist<=epsilon:    
                    # print('dist',dist)
                    seqList.append(i)
                    continue
        f.write(','.join(str(x) for x in seqList)+'\n')
        # print('seqList',seqList)
    f.close()
    poly_cell=gdspy.Cell('tmp')
    poly=gdspy.PolygonSet(polyvertice,1)
    # poly_cell.add(poly)
    xlength=max(corners[:,0])-min(corners[:,0])
    ylength=max(corners[:,1])-min(corners[:,1])
    nX=args.nX
    nY=args.nY
    xsep=max(xlength,ylength)*args.sep
    ysep=xsep
    for i in range(nX): 
        for j in range(nY):
            xpos=(xlength+xsep)*(i+1)
            ypos=(ylength+ysep)*(j+1)
            trans=gdspy.copy(poly,xpos,ypos)
            # print poly
            poly_cell.add(trans)

    if args.out_file is not None: 
        gdspy.write_gds(args.out_file,unit=args.scale*1.0e-9,precision=1.0e-9)

def idx2xy(corners,idx):
    cornerList=[]
    for i in idx:
        cornerList.append(corners[i])
    return cornerList

def verticeAlign(corners):
    print('test')
    return 0

parser=argparse.ArgumentParser(description='This script is used to extract corner points from image')
parser.add_argument('-i',dest='input_image',required=True,help='input image file',type=str)
parser.add_argument('-s',dest='corner_seq',help='image corner sequence',type=str)
parser.add_argument('-nx',dest='nX',default=1,help='replicate nx times along x axis',type=int)
parser.add_argument('-ny',dest='nY',default=1,help='replicate ny times along y axis',type=int)
parser.add_argument('-sep',dest='sep',default=0.1,help='separation ratio',type=float)
parser.add_argument('-scale',dest='scale',default=1.0,help='unit=scale nm',type=float)
parser.add_argument('-o',dest='out_file',help='output file',type=str)
args=parser.parse_args()

corners,ans=image2corners(args)
if (args.out_file is not None): 
    if (args.corner_seq is None):
        print('\tUse auto mode: connect for H-V polygon')
        autoHV(corners,ans)
        print('\tSuccessfully output gds :)')
    else:
        print('\tUse manual mode: input vertice sequence')
        corners2gds(corners,args)
        print('\tSuccessfully output gds :)')
else:
    print("\tWarning: No specify output file, no gds output")
