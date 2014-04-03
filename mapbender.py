#!/usr/bin/env python

import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import tee, izip
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
from PIL import Image

def y2lat(a):
    return 180.0/math.pi*(2.0*math.atan(math.exp(a*math.pi/180.0))-math.pi/2.0)

def lat2y(a):
    return 180.0/math.pi*math.log(math.tan(math.pi/4.0+a*(math.pi/180.0)/2.0))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = tee(iterable, 2)
    next(b, None)
    return izip(a, b)

def triplewise(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
    a,b,c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return izip(a,b,c)

# using barycentric coordinates
def ptInTriangle(p, p0, p1, p2):
    A = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
    sign = -1 if A < 0 else 1;
    s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]) * sign;
    t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]) * sign;
    return s >= 0 and t >= 0 and (s + t) <= 2 * A * sign;

def getxing(p0, p1, p2, p3):
    ux = p1[0]-p0[0]
    uy = p1[1]-p0[1]
    vx = p2[0]-p3[0]
    vy = p2[1]-p3[1]
    # get multiplicity of u at which u meets v
    a = vy*ux-vx*uy
    if a == 0:
        # lines are parallel and never meet
        return None
    s = (vy*(p3[0]-p0[0])+vx*(p0[1]-p3[1]))/a
    if 0.0 < s < 1.0:
        return (p0[0]+s*ux, p0[1]+s*uy)
    else:
        return None

# the line p0-p1 is the upper normal to the path
# the line p2-p3 is the lower normal to the path
#
#  |        |        |
# p0--------|--------p1
#  |        |        |
#  |        |        |
# p3--------|--------p2
#  |        |        |
def ptInQuadrilateral(p, p0, p1, p2, p3):
    # it might be that the two normals cross at some point
    # in that case the two triangles are created differently
    cross = getxing(p0, p1, p2, p3)
    if cross:
        return ptInTriangle(p, p0, cross, p3) or ptInTriangle(p, p2, cross, p1)
    else:
        return ptInTriangle(p, p0, p1, p2) or ptInTriangle(p, p2, p3, p0)

def get_st(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Xx,Xy):
    d = Bx-Ax-Cx+Dx
    e = By-Ay-Cy+Dy
    l = Dx-Ax
    g = Dy-Ay
    h = Cx-Dx
    m = Cy-Dy
    i = Xx-Dx
    j = Xy-Dy
    n = g*h-m*l
    # calculation for s
    a1 = m*d-h*e
    b1 = n-j*d+i*e
    c1 = l*j-g*i
    # calculation for t
    a2 = g*d-l*e
    b2 = n+j*d-i*e
    c2 = h*j-m*i
    s = []
    if a1 == 0:
        s.append(-c1/b1)
    else:
        r1 = b1*b1-4*a1*c1
        if r1 >= 0:
            r11 = (-b1+sqrt(r1))/(2*a1)
            if -0.0000000001 <= r11 <= 1.0000000001:
                s.append(r11)
            r12 = (-b1-sqrt(r1))/(2*a1)
            if -0.0000000001 <= r12 <= 1.0000000001:
                s.append(r12)
    t = []
    if a2 == 0:
        t.append(-c2/b2)
    else:
        r2 = b2*b2-4*a2*c2
        if r2 >= 0:
            r21 = (-b2+sqrt(r2))/(2*a2)
            if -0.0000000001 <= r21 <= 1.0000000001:
                t.append(r21)
            r22 = (-b2-sqrt(r2))/(2*a2)
            if -0.0000000001 <= r22 <= 1.0000000001:
                t.append(r22)
    if not s or not t:
        return [],[]
    if len(s) == 1 and len(t) == 2:
        s = [s[0],s[0]]
    if len(s) == 2 and len(t) == 1:
        t = [t[0],t[0]]
    return s, t

def main(x,y,width,smoothing,subdiv):
    halfwidth = width/2.0
    tck,u = interpolate.splprep([x,y],s=smoothing)
    unew = np.linspace(0,1.0,subdiv+1)
    out = interpolate.splev(unew,tck)
    heights = []
    offs = []
    height = 0.0
    for (ax,ay),(bx,by) in pairwise(zip(*out)):
        s = ax-bx
        t = ay-by
        l = sqrt(s*s+t*t)
        offs.append(height)
        height += l
        heights.append(l)
    # the border of the first segment is just perpendicular to the path
    cx = -out[1][1]+out[1][0]
    cy = out[0][1]-out[0][0]
    cl = sqrt(cx*cx+cy*cy)/halfwidth
    dx = out[1][1]-out[1][0]
    dy = -out[0][1]+out[0][0]
    dl = sqrt(dx*dx+dy*dy)/halfwidth
    px = [out[0][0]+cx/cl]
    py = [out[1][0]+cy/cl]
    qx = [out[0][0]+dx/dl]
    qy = [out[1][0]+dy/dl]
    for (ubx,uby),(ux,uy),(uax,uay) in triplewise(zip(*out)):
        # get adjacent line segment vectors
        ax = ux-ubx
        ay = uy-uby
        bx = uax-ux
        by = uay-uy
        # normalize length
        al = sqrt(ax*ax+ay*ay)
        bl = sqrt(bx*bx+by*by)
        ax = ax/al
        ay = ay/al
        bx = bx/bl
        by = by/bl
        # get vector perpendicular to sum
        cx = -ay-by
        cy = ax+bx
        cl = sqrt(cx*cx+cy*cy)/halfwidth
        px.append(ux+cx/cl)
        py.append(uy+cy/cl)
        # and in the other direction
        dx = ay+by
        dy = -ax-bx
        dl = sqrt(dx*dx+dy*dy)/halfwidth
        qx.append(ux+dx/dl)
        qy.append(uy+dy/dl)
    # the border of the last segment is just perpendicular to the path
    cx = -out[1][-1]+out[1][-2]
    cy = out[0][-1]-out[0][-2]
    cl = sqrt(cx*cx+cy*cy)/halfwidth
    dx = out[1][-1]-out[1][-2]
    dy = -out[0][-1]+out[0][-2]
    dl = sqrt(dx*dx+dy*dy)/halfwidth
    px.append(out[0][-1]+cx/cl)
    py.append(out[1][-1]+cy/cl)
    qx.append(out[0][-1]+dx/dl)
    qy.append(out[1][-1]+dy/dl)
    quads = []
    patches = []
    for (p3x,p3y,p2x,p2y),(p0x,p0y,p1x,p1y) in pairwise(zip(px,py,qx,qy)):
        quads.append(((p0x,p0y),(p1x,p1y),(p2x,p2y),(p3x,p3y)))
        polygon = Polygon(((p0x,p0y),(p1x,p1y),(p2x,p2y),(p3x,p3y)), True)
        patches.append(polygon)
    containingquad = []
    for pt in zip(x,y):
        # for each point, find the quadrilateral that contains it
        found = []
        for i,(p0,p1,p2,p3) in enumerate(quads):
            if ptInQuadrilateral(pt,p0,p1,p2,p3):
                found.append(i)
        if found:
            if len(found) > 1:
                print "point found in two quads"
                return None
            containingquad.append(found[0])
        else:
            containingquad.append(None)
    # check if the only points for which no quad could be found are in the
    # beginning or in the end
    # find the first missing ones:
    for i,q in enumerate(containingquad):
        if q != None:
            break
    # find the last missing ones
    for j,q in izip(xrange(len(containingquad)-1, -1, -1), reversed(containingquad)):
        if q != None:
            break
    # remove the first and last missing ones
    if i != 0 or j != len(containingquad)-1:
        containingquad = containingquad[i:j+1]
        x = x[i:j+1]
        y = y[i:j+1]
    # check if there are any remaining missing ones:
    if None in containingquad:
        print "cannot find quad for point"
        return None
    for off,h in zip(offs,heights):
        targetquad = ((0,off+h),(width,off+h),(width,off),(0,off))
        patches.append(Polygon(targetquad,True))
    tx = []
    ty = []
    assert len(containingquad) == len(x) == len(y)
    assert len(out[0]) == len(out[1]) == len(px) == len(py) == len(qx) == len(qy) == len(quads)+1 == len(heights)+1 == len(offs)+1
    for (rx,ry),i in zip(zip(x,y),containingquad):
        if i == None:
            continue
        (ax,ay),(bx,by),(cx,cy),(dx,dy) = quads[i]
        s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,rx,ry)
        # if more than one solution, take second 
        # TODO: investigate if this is always the right solution
        if len(s) != 1 or len(t) != 1:
            s = s[1]
            t = t[1]
        else:
            s = s[0]
            t = t[0]
        u = s*width
        v = offs[i]+t*heights[i]
        tx.append(u)
        ty.append(v)
    #sx = []
    #sy = []
    #for ((x1,y1),(x2,y2)),((ax,ay),(bx,by),(cx,cy),(dx,dy)),off,h in zip(pairwise(zip(*out)),quads,offs,heights):
    #    s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,x1,y1)
    #    if len(s) != 1 or len(t) != 1:
    #        return None
    #    u = s[0]*width
    #    v = off+t[0]*h
    #    sx.append(u)
    #    sy.append(v)
    #    s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,x2,y2)
    #    if len(s) != 1 or len(t) != 1:
    #        return None
    #    u = s[0]*width
    #    v = off+t[0]*h
    #    sx.append(u)
    #    sy.append(v)
    im = Image.open("map.png")
    bbox = [8.0419921875,51.25160146817652,10.074462890625,54.03681240523652]
    # apply mercator projection
    bbox[1] = lat2y(bbox[1])
    bbox[3] = lat2y(bbox[3])
    iw,ih = im.size
    data = []
    for i,(off,h,(p0,p1,p2,p3)) in enumerate(zip(offs,heights,quads)):
        # first, account for the offset of the input image
        p0 = p0[0]-bbox[0],p0[1]-bbox[1]
        p1 = p1[0]-bbox[0],p1[1]-bbox[1]
        p2 = p2[0]-bbox[0],p2[1]-bbox[1]
        p3 = p3[0]-bbox[0],p3[1]-bbox[1]
        # PIL expects coordinates in counter clockwise order
        p1,p3 = p3,p1
        #   x          lon
        # ----- =     -----
        #   w     bbox[2]-bbox[0]
        # translate to pixel coordinates
        p0 = (iw*p0[0])/(bbox[2]-bbox[0]),(ih*p0[1])/(bbox[3]-bbox[1])
        p1 = (iw*p1[0])/(bbox[2]-bbox[0]),(ih*p1[1])/(bbox[3]-bbox[1])
        p2 = (iw*p2[0])/(bbox[2]-bbox[0]),(ih*p2[1])/(bbox[3]-bbox[1])
        p3 = (iw*p3[0])/(bbox[2]-bbox[0]),(ih*p3[1])/(bbox[3]-bbox[1])
        # PIL starts coordinate system at the upper left corner, swap y coord
        p0 = int(p0[0]),int(ih-p0[1])
        p1 = int(p1[0]),int(ih-p1[1])
        p2 = int(p2[0]),int(ih-p2[1])
        p3 = int(p3[0]),int(ih-p3[1])
        box=(0,int(ih*(height-off-h)/(bbox[3]-bbox[1])),
                int(iw*width/(bbox[2]-bbox[0])),int(ih*(height-off)/(bbox[3]-bbox[1])))
        quad=(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],p3[0],p3[1])
        data.append((box,quad))
    im_out = im.transform((int(iw*width/(bbox[2]-bbox[0])),int(ih*height/(bbox[3]-bbox[1]))),Image.MESH,data,Image.BICUBIC)
    im_out.save("out.png")
    np.random.seed(seed=0)
    colors = 100*np.random.rand(len(patches)/2)+100*np.random.rand(len(patches)/2)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    p.set_array(np.array(colors))
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.axhspan(0, height, xmin=0, xmax=width)
    fig, ax = plt.subplots()
    ax.add_collection(p)
    ax.set_aspect('equal')
    plt.imshow(np.asarray(im_out),extent=[0,width,0,height])
    plt.imshow(np.asarray(im),extent=[bbox[0],bbox[2],bbox[1],bbox[3]])
    plt.plot(x,y,out[0],out[1],px,py,qx,qy,tx,ty)
    plt.show()
    return True

if __name__ == '__main__':
    x = []
    y = []
    import sys
    if len(sys.argv) != 5:
        print "usage: %s data.csv width smoothing N"%sys.argv[0]
        print ""
        print " data.csv     whitespace delimited lon/lat pairs of points along the path"
        print " width        width of the resulting map in degrees"
        print " smoothing    curve smoothing from 0 (exact fit) to higher values (looser fit)"
        print " N            amount of quads to split the path into"
        print ""
        print " example usage:"
        print "              %s Weser-Radweg-Hauptroute.csv 0.286 6 20"%sys.argv[0]
        exit(1)
    with open(sys.argv[1]) as f:
        for l in f:
            a,b = l.split()
            # apply mercator projection
            b = lat2y(float(b))
            x.append(float(a))
            y.append(b)
    width = float(sys.argv[2])
    smoothing = float(sys.argv[3])
    N = int(sys.argv[4])
    main(x,y,width,smoothing,N)
    #for smoothing in [1,2,4,8,12]:
    #    for subdiv in range(10,30):
    #        if main(x,y,width,smoothing,subdiv):
    #            print width,smoothing,subdiv
