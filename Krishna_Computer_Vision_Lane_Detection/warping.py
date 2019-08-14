# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import cv2

def persp_warp(img,  
               src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
               dest=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    dest_size = (img.shape[1], img.shape[0])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dest = dest * np.float32(dest_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dest_size)
    return warped