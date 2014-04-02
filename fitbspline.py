#!/usr/bin/env python

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import tee, izip
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib

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
    if s < 1.0:
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
    #if cross:
    #    return ptInTriangle(p, p0, cross, p3) or ptInTriangle(p, p2, cross, p1)
    #else:
    #    return ptInTriangle(p, p0, p1, p2) or ptInTriangle(p, p2, p3, p0)
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
        return None
    if len(s) == 1 and len(t) == 2:
        s = [s[0],s[0]]
    if len(s) == 2 and len(t) == 1:
        t = [t[0],t[0]]
    return s, t

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    #res = np.dot(np.linalg.inv(A), B)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def main():
    width = 2/5.0
    halfwidth = width/2.0
    x = []
    y = []
    with open(sys.argv[1]) as f:
        for l in f:
            a,b = l.split()
            x.append(float(a))
            y.append(float(b))
    tck,u = interpolate.splprep([x,y],s=10)
    unew = np.arange(0,1.1,0.1)
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
    #for (p0x,p0y,p1x,p1y),(p3x,p3y,p2x,p2y) in pairwise(zip(px,py,qx,qy)):
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
            if len(found) > 2:
                print found
            containingquad.append(found)
        else:
            print "can't find quad for point"
            containingquad.append(None)
            #exit(1)
    print containingquad
    trans = []
    print width, height
    srcquads = []
    for off,h,srcquad in zip(offs,heights,quads):
        #targetquad = ((0,height-off),(width,height-off),(width,height-off-h),(0,height-off-h))
        targetquad = ((0,off+h),(width,off+h),(width,off),(0,off))
        trans.append(find_coeffs(srcquad,targetquad))
        patches.append(Polygon(targetquad,True))
    tx = []
    ty = []
    #targetquad = (0,height),(width,height),(width,0),(0,0)
    #srcquad = (min(x),max(y)),(max(x),max(y)),(max(x),min(y)),(min(x),min(y)) 
    #trans = find_coeffs(srcquad,targetquad)
    #for (rx,ry) in zip(x,y):
    #    a,b,c,d,e,f,g,h = trans
    #    u = (a*rx + b*ry + c)/(g*rx + h*ry + 1)
    #    v = (d*rx + e*ry + f)/(g*rx + h*ry + 1)
    #    tx.append(u)
    #    ty.append(v)
    assert len(containingquad) == len(x) == len(y)
    assert len(out[0]) == len(out[1]) == len(px) == len(py) == len(qx) == len(qy) == len(quads)+1 == len(heights)+1 == len(offs)+1 == len(trans)+1
    for (rx,ry),l in zip(zip(x,y),containingquad):
        if not l:
            continue
        for i in l[:1]:
            (ax,ay),(bx,by),(cx,cy),(dx,dy) = quads[i]
            s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,rx,ry)
            if len(s) != 1 or len(t) != 1:
                print "fail"
                exit(1)
            #a,b,c,d,e,f,g,h = trans[i]
            ##den = -a*e+a*h*ry+b*d-b*g*ry-d*h*rx+e*g*rx
            ##tx.append((-b*f+b*ry+c*e-c*h*ry-e*rx+f*h*rx)/den)
            ##ty.append((a*f-a*ry-c*d+c*g*ry+d*rx-f*g*rx)/den)
            #u = (a*rx + b*ry + c)/(g*rx + h*ry + 1)
            #v = (d*rx + e*ry + f)/(g*rx + h*ry + 1)
            u = s[0]*width
            v = offs[i]+t[0]*heights[i]
            tx.append(u)
            ty.append(v)
    sx = []
    sy = []
    for ((x1,y1),(x2,y2)),((ax,ay),(bx,by),(cx,cy),(dx,dy)),off,h in zip(pairwise(zip(*out)),quads,offs,heights):
        s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,x1,y1)
        if len(s) != 1 or len(t) != 1:
            print "fail"
            exit(1)
        #u = (a*ax + b*ay + c)/(g*ax + h*ay + 1)
        #v = (d*ax + e*ay + f)/(g*ax + h*ay + 1)
        u = s[0]*width
        v = off+t[0]*h
        sx.append(u)
        sy.append(v)
        #u = (a*bx + b*by + c)/(g*bx + h*by + 1)
        #v = (d*bx + e*by + f)/(g*bx + h*by + 1)
        s,t = get_st(ax,ay,bx,by,cx,cy,dx,dy,x2,y2)
        if len(s) != 1 or len(t) != 1:
            print "fail"
            exit(1)
        u = s[0]*width
        v = off+t[0]*h
        sx.append(u)
        sy.append(v)
    colors = 100*np.random.rand(len(patches)/2)+100*np.random.rand(len(patches)/2)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    p.set_array(np.array(colors))
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.axhspan(0, height, xmin=0, xmax=width)
    fig, ax = plt.subplots()
    ax.add_collection(p)
    ax.set_aspect('equal')
    plt.plot(x,y,out[0],out[1],px,py,qx,qy,sx,sy,tx,ty)
    #plt.plot(tx,ty)
    plt.show()

if __name__ == '__main__':
    main()
