
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import cv
import numpy as np

def im2array(im,dtype=None):
    """a = im2array(im) converts ipl im to numpy.array a"""
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

    if dtype == None:
      dtype = depth2dtype[im.depth]
    a = np.fromstring(
         im.tostring(),
         dtype=dtype,
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height,im.width,im.nChannels)
    return a

def array2im(a):
    """im = array2im(a) converts numpy.array a to ipl im"""
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
                                   dtype2depth[str(a.dtype)], nChannels)
    cv.SetData(cv_im, a.tostring(),a.dtype.itemsize*nChannels*a.shape[1])
    return cv_im 
      
def array2cv(a):
    """mat = array2cv(a) converts array a to cvMat mat"""
    return cv.GetMat(array2im(np.array(a)))

def cv2array(mat):
    """a = cv2array(mat) converts cvMat mat to numpy.array a"""
    return im2array(cv.GetImage(mat))


def rodrigues(r):
    """R = rodrigues  converts a rot mat to rot vec or vice-versa"""
    rcv = array2cv(r)

    if np.array(r.shape).min() == 1:
        Rcv = cv.CreateMat(3, 3, cv.CV_64FC1)
    else:
        Rcv = cv.CreateMat(1, 3, cv.CV_64FC1)

    cv.Rodrigues2(rcv,Rcv)

    return cv2array(Rcv)[...,0]

def adaptiveThreshold(im, max, amethod=cv.CV_ADAPTIVE_THRESH_MEAN_C,
                      ttype=cv.CV_THRESH_BINARY, bsize=3, param=5):
    """
    jm = adaptiveThreshold
    Executes cv.AdaptiveThreshold on arrays;
    see that function's defn for details

    INPUTS
      im - np.array - input image

    OUTPUTS
      jm - np.array - thresholded image
    """
    imcv = array2cv(np.array(255*im,dtype=np.uint8))
    jmcv = array2cv(np.array(255*im,dtype=np.uint8))

    cv.AdaptiveThreshold(imcv, jmcv, max, amethod, ttype, bsize, param)

    jm = np.array(cv2array(jmcv)[...,0],dtype=float)

    return jm
 
def smooth(im, stype=cv.CV_GAUSSIAN, p1=3, p2=0, p3=0, p4=0):
    """
    jm = smooth
    Executes cv.Smooth on arrays;
    see that function's defn for details

    INPUTS
      im - np.array - input image

    OUTPUTS
      jm - np.array - smoothed image
    """
    imcv = array2cv(np.array(255*im,dtype=np.uint8))
    jmcv = array2cv(np.array(255*im,dtype=np.uint8))

    cv.Smooth(imcv, jmcv, stype, p1, p2, p3, p4)

    jm = np.array(cv2array(jmcv)[...,0],dtype=float)/255.

    return jm
 
def filter2d(im, ker):
    """
    jm = filter2d
    Executes cv.Filter2D on array;
    see that function's defn for details

    INPUTS
      im - np.array - input image
      ker - np.array - convolution kernel

    OUTPUTS
      jm - np.array - filtered image
    """
    imcv  = array2cv(im)
    jmcv  = array2cv(im)
    kercv = array2cv(ker)

    cv.Filter2D(imcv, jmcv, kercv)

    jm = cv2array(jmcv)[...,0]

    return jm

def findCB(srcs, sq=(6,8), sc=30.0, viz=False, vb=False):
    """
    frs, cb, px = findCB(srcs, sq, sc, viz)  finds chessboard corners

    INPUTS
      srcs - list - framesource pipe / plugin chain
      sq - tuple - (#cols, #rows) in chessboard pattern
      sc - float - dimension of one chessboard square
      viz - bool - whether to show chessboard corners when found
      vb - bool - whether to print debugging information

    OUTPUTS
      frs - 1 x Nf - frames where chessboard corners were found
      cb - Nc x 3 - 3D chessboard corner locations
      pxs - len-Nf list of Nc x 2 - pixel locations of corners in each frame
    """
    import src
    N = src.info(srcs).N

    K,L = np.meshgrid(np.arange(sq[0])-sq[0]/2.0,np.arange(sq[1])-sq[1]/2.0)
    cb = sc*np.array([(k,l,0.0) for k,l in zip(K.flatten(),L.flatten())])
    
    frs = np.zeros((0))
    pxs = []

    for n in range(N):

        im = src.getIm(srcs,n)
        fr = array2cv(im)
        found, pts0 = cv.FindChessboardCorners(fr, sq)

        if vb:
            if not found:
                print '#%d:  chessboard not found' % n
                continue

            print '%d' % n

        if cv.GetImage(fr).channels == 3:
          gr = cv.CreateImage(cv.GetSize(fr),8,1)
          cv.CvtColor(fr,gr,cv.CV_RGB2GRAY)
        else:
          gr = fr

        pts = cv.FindCornerSubPix(gr, pts0, (11,11), (-1,-1), 
                 (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,30,0.01))
        #pts = pts0

        px = np.array(pts)

        if viz:
            import pylab as plt
            #cv.DrawChessboardCorners(fr, sq, pts, found)
            im = cv2array(fr)
            plt.figure(1); plt.clf()
            ax = plt.imshow(np.array(im))#,cmap=plt.cm.gray)
            ax.set_interpolation('nearest')
            plt.plot(px[:,0],px[:,1],'r+')
            plt.axis('image')
            plt.show()

        frs = np.hstack((frs,n))
        pxs += [px]

    return frs, cb, pxs

def calCam(xyz, px, cnt, imgsz, 
           Acv=cv.CreateMat(3, 3, cv.CV_64FC1),
           dcv=cv.CreateMat(1, 4, cv.CV_64FC1),
           flags=0):
    """A, d, r, t, R = calCam
    Executes cv.CalibrateCamera2 on arrays, returning
    camera paramters and reprojection error in arrays.

    INPUTS
      xyz - N x 3 - 3D calibration object coordinates
      px  - N x 2 - corresponding 2D pixel observations
      cnt - 1 x M - partition of N observations into M views
      imsz - 1 x 2 - image size (Ncols, Nrows)

    OUTPUTS
      A - 3 x 3 - intrinsic camera parameters
      d - 1 x 4 - distortion coefficients
      r - M x 3 - rotation (vector)
      t - M x 3 - translation
      R - 3 x 3 x M - rotation (matrix)
    """
    ptcnt = [len(xyz)]

    M = len(cnt) 

    rcv = cv.CreateMat(M, 3, cv.CV_64FC1)
    Rcv = cv.CreateMat(3, 3, cv.CV_64FC1)
    tcv = cv.CreateMat(M, 3, cv.CV_64FC1)

    objpts = array2cv(xyz)
    imgpts = array2cv(px)
    pntcnt = array2cv(cnt)

    cv.CalibrateCamera2(objpts, imgpts, pntcnt, imgsz, 
                        Acv, dcv, rcv, tcv, flags)

    A = cv2array(Acv)[...,0]
    d = cv2array(dcv)[...,0]
    r = cv2array(rcv)[...,0]
    t = cv2array(tcv)[...,0]
    R = np.zeros((3,3,M))
    for m in range(M):
        cv.Rodrigues2(rcv[m,:],Rcv)
        R[...,m] = cv2array(Rcv)[...,0]

    return A, d, r, t, R

def calCams(xyz, px1, px2, cnt, imgsz, A1, d1, A2, d2):
    """R, t = calCams
    Executes cv.StereoCalibrate on arrays, returning
    A_i, d_i, R, t in arrays.

    INPUTS
      xyz - N x 3 - 3D calibration object coordinates
      px_i  - N x 2 - corresponding 2D pixel observations
      cnt - 1 x M - partition of N observations into M views
      imsz - 1 x 2 - image size (Ncols, Nrows)
      Ai, di - intrinsic camera parameters

    OUTPUTS
      R - 3 x 3 - rotation between camera coord systems
      t - 3 x 1 - translation between camera coord systems
    """
    Rcv = cv.CreateMat(3, 3, cv.CV_64FC1)
    tcv = cv.CreateMat(3, 1, cv.CV_64FC1)

    obj = array2cv(xyz)
    img1 = array2cv(px1)
    img2 = array2cv(px2)
    pntcnt = array2cv([cnt])

    A1cv = array2cv(A1)
    d1cv = array2cv(d1)
    A2cv = array2cv(A2)
    d2cv = array2cv(d2)

    # Calibrate two cameras together
    cv.StereoCalibrate(obj, img1, img2, pntcnt, 
                       A1cv, d1cv, A2cv, d2cv, imgsz, Rcv, tcv,
                       #flags=cv.CV_CALIB_USE_INTRINSIC_GUESS |
                       #      cv.CV_CALIB_FIX_PRINCIPAL_POINT)
                       flags=cv.CV_CALIB_FIX_INTRINSIC)

    R  = cv2array(Rcv)[...,0]
    t  = cv2array(tcv)[...,0]

    return R, t

def findExt(xyz, px, A, d, cnt=np.array([])):
    """R, t = findExt
    Executes cv.FindExtrinsicCameraParams2 on multiple poses

    INPUTS
      xyz - N x 3 - 3D calibration object coordinates
      px  - N x 2 - corresponding 2D pixel observations
      A - 3 x 3 - intrinsic camera parameters
      d - 1 x 4 - distortion coefficients
      cnt - 1 x M - partition of N observations into M views
                    (defaults to [N])

    OUTPUTS
      R - 3 x 3 x M - rotation (matrix)
      t - M x 3 - translation
    """
    if cnt.shape[0] == 0:
        cnt = [xyz.shape[0]]

    M = len(cnt) 

    R = np.kron(np.ones((3,3,M)),np.nan)
    t = np.kron(np.ones((M,3)),  np.nan)

    Acv = array2cv(A)
    dcv = array2cv(d.flatten()[:,np.newaxis])
    rcv = cv.CreateMat(1, 3, cv.CV_64FC1)
    tcv = cv.CreateMat(1, 3, cv.CV_64FC1)

    k = 0
    for m in range(M):
        xyzcv = array2cv(xyz[k:k+cnt[m],:])
        pxcv  = array2cv(px [k:k+cnt[m],:])

        cv.FindExtrinsicCameraParams2(xyzcv, pxcv, Acv, dcv, rcv, tcv)
        R[...,m] = rodrigues(cv2array(rcv)[...,0])
        t[m,:]   = cv2array(tcv)[...,0]

        k = k + cnt[m]

    return R, t


def pxError(X,p,cnt,A,d,R,t):
    """
    err = pxError  compute pixel error for calibration
    """
    err = np.zeros((2,0))
    q = np.zeros((2,0))
    d = d.flatten()
    cnt = cnt.flatten()
    c = 0
    for i in range(len(cnt)):
        ones = np.ones((1,cnt[i]))
        x = np.dot(R[...,i],X[:,c:c+cnt[i]])+np.kron(ones,t[:,i:i+1])
        xp = x[0:1,:] / x[2,:]
        yp = x[1:2,:] / x[2,:]
        r2 = xp**2 + xp**2
        xpp = xp*(1+d[0]*r2+d[1]*r2**2) + 2*d[2]*xp*yp + d[3]*(r2+2*xp**2)
        ypp = yp*(1+d[0]*r2+d[1]*r2**2) + d[2]*(r2+2*yp**2) + 2*d[3]*xp*yp 
        q = np.hstack((q,np.dot(A,np.vstack((xpp,ypp,np.ones((1,xpp.size)))))[0:2,:]))
        c = c+cnt[i]

    err = np.hstack((err,p-q))

    return err

def mmError(X,R1,t1,R2,t2,R12,t12):
    """
    err, X1, X2 = mmError  compute millimeter error for calibration
    """
    err = np.zeros((3,0))
    X1 = np.zeros((3,0))
    X2 = np.zeros((3,0))
    for i in range(t1.shape[1]):
        ones = np.ones((1,X.shape[1]))
        z1 = np.dot(R2[...,i],X)+np.kron(ones,t2[:,i:i+1])
        z2 = (np.dot(R12,np.dot(R1[...,i],X))
             +np.kron(ones,np.dot(R12,t1[:,i:i+1])+t12))
        err = np.hstack((err,z1-z2))
        X1 = np.hstack((X1,z1))
        X2 = np.hstack((X2,z2))

    return err, X1, X2


