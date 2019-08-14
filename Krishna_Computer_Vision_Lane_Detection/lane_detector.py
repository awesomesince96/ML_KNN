# -*- coding: utf-8 -*-
"""
Computer Vision: Lane Detection
Binit Gajera - KH20736
Krishna Prajapati - PI87410
"""

import sys
import argparse
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from undistort import CameraCalibration, create_pipeline
from warping import persp_warp
from window import slide_window
from curve_lane import draw_lanes 
    
def lane_detector(img):    
    cal_cam = CameraCalibration('camera_cal/*.jpg', 'camera_cal/cal_pickle.p')    
    img = cal_cam.undistort(img)
    img_t = create_pipeline(img)
    img_t = persp_warp(img_t)
    out_img, curves, lanes, ploty = slide_window(img_t, draw_windows=False)
    img = draw_lanes(img, curves[0], curves[1])
    return img

def visualise(vid, vis_frame):
    img_frame = vid.get_frame(vis_frame)
    img = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    cal_cam = CameraCalibration('camera_cal/*.jpg', 'camera_cal/cal_pickle.p')    
    img = cal_cam.undistort(img)
    dest = create_pipeline(img)
    dest = persp_warp(dest)
    out_img, curves, lanes, ploty = slide_window(dest)
    img_final = draw_lanes(img, curves[0], curves[1])
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 2))
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=10)
    ax2.imshow(dest)
    ax2.set_title('Filter+Perspective', fontsize=10)
    ax2.axis('off')
    ax3.imshow(out_img)
    ax3.plot(curves[0], ploty, color='yellow', linewidth=3)
    ax3.plot(curves[1], ploty, color='yellow', linewidth=3)
    ax3.set_title('Sliding window+Curve Fit', fontsize=10)
    ax3.axis('off')
    ax4.imshow(img_final)
    ax4.set_title('Overlay Lanes', fontsize=10)
    ax4.axis('off')
    plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
    plt.show()

def main(args):
    parser = argparse.ArgumentParser(
        description='Detecting lane from a dashcam video')
    
    parser.add_argument('-i', '--input-video', required=True, type=str, help='Path to the input video')
    parser.add_argument('-v', '--visualisation', required=False, type=int, help='Visualise backend components of a specific frame at t timestep')
    args = parser.parse_args(args)
    car_vid = VideoFileClip(args.input_video)#.subclip(40,43)
    output_name = 'output.mp4'
    cal_cam = CameraCalibration('camera_cal/*.jpg', 'camera_cal/cal_pickle.p')    
    cal_cam.undistort_img()
    car_vid = car_vid.resize((1280, 720))
    f_vid = car_vid.fl_image(lane_detector)
    f_vid.write_videofile(output_name, audio=False)
    if args.visualisation is not None:
        if args.visualisation <= car_vid.duration:
            visualise(car_vid, args.visualisation)
        else:
            print("Please input timestep less than video duration!!")

if __name__ == "__main__":
    main(sys.argv[1:])