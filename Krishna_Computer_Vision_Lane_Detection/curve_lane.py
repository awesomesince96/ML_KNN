# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import cv2

def find_curve(img, leftx, rightx):
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(plot_y)
    per_pix_y = 30.5/720 # meters per pixel in y dimension
    per_pix_x = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y*per_pix_y, leftx*per_pix_x, 2)
    right_fit_cr = np.polyfit(plot_y*per_pix_y, rightx*per_pix_x, 2)
    # Calculate the new radii of curvature
    l_curve_rad = ((1 + (2*left_fit_cr[0]*y_eval*per_pix_y + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    r_curve_rad = ((1 + (2*right_fit_cr[0]*y_eval*per_pix_y + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    left_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    right_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (right_fit_x_int + left_fit_x_int) /2
    center = (car_pos - lane_center_position) * per_pix_x / 10
    # Now our radius of curvature is in meters
    return (l_curve_rad, r_curve_rad, center)

def inverse_persp_warp(img, 
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dest=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dest_size = (img.shape[1], img.shape[0])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dest = dest * np.float32(dest_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dest_size)
    return warped

def draw_lanes(img, left_fit, right_fit):
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    l = np.array([np.transpose(np.vstack([left_fit, plot_y]))])
    r = np.array([np.flipud(np.transpose(np.vstack([right_fit, plot_y])))])
    points = np.hstack((l, r))
    
    cv2.fillPoly(color_img, np.int_(points), (255, 160, 60))
    inv_perspective = inverse_persp_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective