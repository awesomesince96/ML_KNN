# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pickle
import cv2
import glob

class CameraCalibration:
    def __init__(self, cal_img, cal_dir):
        self.cal_img = cal_img
        self.cal_dir = cal_dir
        
    def undistort_img(self):
        # Prepare object points 0,0,0 ... 8,5,0
        obj_pts = np.zeros((6*9,3), np.float32)
        obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Stores all object points & img points from all images
        objpoints = []
        imgpoints = []

        # Get directory for all calibration images
        images = glob.glob(self.cal_img)

        for indx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret == True:
                objpoints.append(obj_pts)
                imgpoints.append(corners)
                # Test undistortion on img
        img_size = (img.shape[1], img.shape[0])

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

        # Save camera calibration for later use
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump( dist_pickle, open(self.cal_dir, 'wb') )

    def undistort(self, img):
        with open(self.cal_dir, mode='rb') as f:
            file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']
        dst = cv2.undistort(img, mtx, dist, None, mtx)
            
        return dst
    
def create_pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]    
    # Sobel x
    sobel_filter = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobel_filter) # Absolute x derivative to accentuate lines away from horizontal
    scal_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scal_sobel)
    sxbinary[(scal_sobel >= sx_thresh[0]) & (scal_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary