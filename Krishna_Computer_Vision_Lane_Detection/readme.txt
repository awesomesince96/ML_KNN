To run the code, we will first keep the camera_cal folder in the same directory as the main code.
All other Python files should also be kept in the same main directory.

Then we will run the lane_detector.py file using the following command:
	python lane_detector.py -i ./input_video.mp4 -v 39

In the command, the -i option takes the path to the input video as value.
The -v option is optional and is used for visualising the background steps at a given timestep of the video.
Do note that input to -v is a timestep to video and so it should less than the length of the video(in seconds).